from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer


def _get_cosine_sim(
    candidate: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scale-stable cosine(candidate, gradient) and whether g is nonzero.

    A zero fresh gradient has no direction, so callers use ``has_gradient`` to retain their prior coefficient.
    """
    candidate_vec = candidate.view(-1)
    gradient_vec = gradient.view(-1)
    gradient_norm = torch.linalg.vector_norm(gradient_vec)

    cosine = torch.cosine_similarity(
        candidate_vec,
        gradient_vec,
        dim=0,
        eps=1e-8,
    ).clamp(-1.0, 1.0)
    has_gradient = gradient_norm > 0.0
    return cosine, has_gradient


class MAL_SGD(Optimizer):
    r"""Memory-ALigned heavy-ball SGD with optional Nesterov correction.

    Let :math:`m_t` be the existing momentum buffer and :math:`g_t` the current (possibly
    L2-regularized) gradient. MAL probes the ordinary candidate

    :math:`\hat{m_t} = \beta_{probe} \, m_t + g_t`

    and maps its cosine with :math:`g_t` to retention :math:`r = (1 + \text{cosine}) / 2`.
    Static MAL uses :math:`c = \beta \, r`. Adaptive MAL uses :math:`c = r` and stores it
    as the next probe coefficient (:math:`\beta_{probe} = c`). The committed state is :math:`m_{t+1} \gets c \, m_t + g_t \,`.

    Under Nesterov, the applied direction is :math:`g_t + c \, m_t` using the *updated*
    buffer. Thus, static MAL exactly recovers PyTorch SGDN whenever ``r == 1``;
    adaptive MAL is deliberately a time-varying-coefficient Nesterov variant.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        c: float = 1.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and beta <= 0.0:
            raise ValueError("Nesterov momentum requires a positive beta")

        self.c = c

        decay_params: list[torch.nn.Parameter] = []
        decay_momentum = []
        decay_beta = []

        no_decay_params: list[torch.nn.Parameter] = []
        no_decay_momentum = []

        if weight_decay == 0:
            for p in params:
                if not p.requires_grad:
                    continue

                no_decay_params.append(p)
                no_decay_momentum.append(torch.zeros_like(p))

        else:
            for p in params:
                if not p.requires_grad:
                    continue
                # Exclude biases and 1D normalization parameters from weight decay
                if p.ndim <= 1:
                    no_decay_params.append(p)
                    no_decay_momentum.append(torch.zeros_like(p))
                else:
                    decay_params.append(p)
                    decay_momentum.append(torch.zeros_like(p))
                    decay_beta.append(torch.full((p.shape[0], 1), beta))

        if not decay_params and not no_decay_params:
            raise ValueError("SGD received no trainable parameters.")

        optim_groups = []

        if no_decay_params:
            optim_groups.append(
                {
                    "params": no_decay_params,
                    "momentum": no_decay_momentum,
                    "weight_decay": 0.0,
                    "beta": [p.new_tensor(0.9) for p in no_decay_params]
                }
            )
        if decay_params:
            optim_groups.append(
                {
                    "params": decay_params,
                    "momentum": decay_momentum,
                    "weight_decay": weight_decay,
                    "beta": decay_beta
                },
            )

        defaults = {
            "lr": lr,
            "nesterov": nesterov,
        }  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

    @torch.no_grad()
    def step(self, closure: Callable[[], float | torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            group_beta = group["beta"]
            nesterov = group["nesterov"]

            for i, (p, m) in enumerate(zip(group["params"], group["momentum"])):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    # Coupled weight decay without mutating "p.grad" in-place.
                    g = g.add(p, alpha=wd)

                # Probe the ordinary momentum candidate before committing the MAL edit.
                beta_probe = group_beta[i]  # per-param adaptive scalar tensor
                m_hat = torch.addcmul(g, m, beta_probe)

                if g.ndim > 1:
                    dims = tuple(range(1, g.ndim))
                    denom = (torch.linalg.vector_norm(m_hat, dim=dims, keepdim=True) * torch.linalg.vector_norm(g, dim=dims, keepdim=True)).clamp_min(1e-8)
                    cosine_sim = ((m_hat * g).sum(dims, keepdim=True) / denom).clamp(-1.0, 1.0)
                else:
                    denom = (m_hat.norm() * g.norm()).clamp_min(1e-8)
                    cosine_sim = ((m_hat.view(-1) @ g.view(-1)) / denom).clamp(-1.0, 1.0)

                d = (1.0 - cosine_sim) * 0.5  # normalized cosine distance
                retention = 1.0 - d  # per-output-unit memory-aligned retention

                # Effective momentum coefficient for this step
                beta = beta_probe.lerp_(retention, weight=self.c)

                m.mul_(beta).add_(g)

                if nesterov:
                    # NAG look-ahead with the SAME aligned coefficient that decayed the
                    # memory: u = g + c*m, so perfect alignment (d=0) recovers vanilla
                    # Nesterov exactly, while a suspect memory gets less look-ahead
                    p.sub_(g, alpha=lr)
                    p.addcmul_(m, beta, value=-lr)
                else:
                    p.sub_(m, alpha=lr)

        return loss


class MAL_ADAMW(Optimizer):
    """Memory-ALigned AdamW for transformer training (ViT / LLM).
    The alignment gate modulates ONLY the first moment's EMA coefficient, per
    parameter tensor.
    Probe: the candidate EMA m_hat = beta1*m + (1-beta1)*g vs the fresh gradient;
    effective coefficient c = 1-d (adaptive, stored per layer) or beta1*(1-d) (static).
    The committed EMA is m <- c*m + (1-c)*g. The second moment (scale tracker,
    beta2) and its bias correction are standard AdamW and are never gated.

    First-moment bias correction is exact under dynamic per-tensor c: the running
    product bc_prod = prod_s c_s is tracked per parameter and the correction is
    1 - bc_prod (reduces to 1 - beta1^t for constant c). Adaptive c is capped at
    1 - 1e-4 because c = 1 is the one degenerate EMA value (zero mass on g freezes
    the memory and zeroes the correction). A perfectly-aligned first adaptive step
    has the same bias-corrected direction as AdamW, but deliberately a much longer
    raw first-moment horizon.

    Weight decay is decoupled (AdamW) and never enters the alignment signal.
    LayerNorm gains and biases (ndim <= 1) keep a fixed beta1 unless align_1d=True:
    cosine similarity on small tensors can be noise-dominated.

    Second-moment step counts are per parameter and serialized, so intermittent
    gradients and checkpoint resume retain exact AdamW bias correction.
    """

    MAX_BETA1 = 1.0 - 1e-4

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        adaptive: bool = True,
        align_1d: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        beta1 = betas[0]
        self.adaptive = adaptive
        self.align_1d = align_1d

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for p in params:
            if not p.requires_grad:
                continue
            # Exclude biases and 1D normalization parameters from weight decay
            if weight_decay == 0 or p.ndim <= 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        if not decay_params and not no_decay_params:
            raise ValueError("AdamW received no trainable parameters.")

        optim_groups = []

        for group_params, group_wd in (
            (no_decay_params, 0.0),
            (decay_params, weight_decay),
        ):
            if group_params:
                optim_groups.append(
                    {
                        "params": group_params,
                        "m": [torch.zeros_like(p) for p in group_params],
                        "v": [torch.zeros_like(p) for p in group_params],
                        "weight_decay": group_wd,
                        "beta": [p.new_tensor(beta1) for p in group_params],
                        "bc_prod": [p.new_tensor(1.0) for p in group_params],
                        "step": [0 for _ in group_params],
                    }
                )

        defaults = {"lr": lr, "beta2": betas[1], "eps": eps}
        super().__init__(optim_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float | torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta2 = group["beta2"]
            eps = group["eps"]
            betas = group["beta"]
            bc_prods = group["bc_prod"]
            steps = group["step"]

            for i, (p, m, v) in enumerate(zip(group["params"], group["m"], group["v"])):
                if p.grad is None:
                    continue

                g = p.grad
                steps[i] += 1
                bc2_sqrt = (1.0 - beta2 ** steps[i]) ** 0.5

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay; never enters the gate

                beta1 = betas[i]
                if p.ndim > 1 or self.align_1d:
                    # Alignment of the candidate EMA with the fresh gradient:
                    m_hat = torch.lerp(g, m, beta1)  # = beta1*m + (1-beta1)*g
                    cosine_sim, has_gradient = _get_cosine_sim(m_hat, g)
                    d = (1.0 - cosine_sim) * 0.5  # normalized cosine distance
                    retention = 1.0 - d

                    if self.adaptive:
                        proposed_c = retention.clamp_max(self.MAX_BETA1)
                        c = torch.where(has_gradient, proposed_c, beta1)
                        beta1.copy_(c)
                    else:
                        c = torch.where(has_gradient, beta1 * retention, beta1)
                else:
                    c = beta1  # tiny 1-D params: plain EMA

                m.lerp_(g, 1.0 - c)  # m <- c*m + (1-c)*g
                bc_prods[i].mul_(c)
                bc_prod = bc_prods[i]
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # AdamW update with exact first-moment bias correction under dynamic c
                u = m.div(v.sqrt().div_(bc2_sqrt).add_(eps))
                u.div_(1.0 - bc_prod)
                p.add_(u, alpha=-lr)

        return loss
