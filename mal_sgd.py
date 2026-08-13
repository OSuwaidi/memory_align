from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer


def _get_cosine_sim(
    momentum_or_candidate: torch.Tensor,
    gradient: torch.Tensor,
    momentum_scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate/gradient cosine similarity and gradient non-zero status.

    When ``momentum_scale`` is omitted, ``momentum_or_candidate`` is the already
    materialized candidate. When it is supplied, the candidate is
    ``momentum_scale * momentum_or_candidate + gradient``. Whole-tensor
    alignment expands that affine candidate into three vector dot products so
    it never needs to be materialized.

    A zero fresh gradient has no direction, so callers use ``has_gradient`` to select their configured fallback coefficient.

    The whole-tensor expansion can lose precision if the candidate nearly
    vanishes through cancellation. The final clamp prevents a negative value
    from roundoff; ordinary optimizer states stay well away from the regime
    where this formulation would become inaccurate.
    """
    if momentum_scale is None:
        candidate_vec = momentum_or_candidate.flatten()
        gradient_vec = gradient.flatten()
        gradient_norm = torch.linalg.vector_norm(gradient_vec)
        cosine = torch.cosine_similarity(candidate_vec, gradient_vec, dim=0, eps=1e-8).clamp(-1.0, 1.0)
        return cosine, gradient_norm > 0.0

    momentum_vec = momentum_or_candidate.flatten()
    gradient_vec = gradient.flatten()
    dot = momentum_vec @ gradient_vec
    momentum_sq = momentum_vec @ momentum_vec
    gradient_sq = gradient_vec @ gradient_vec

    numerator = momentum_scale * dot + gradient_sq
    candidate_sq = (momentum_scale**2 * momentum_sq + 2.0 * momentum_scale * dot + gradient_sq).clamp_min(0.0)
    # Match cosine_similarity's per-vector epsilon. Clamping the product instead
    # would suppress valid cosines whenever both norms are small but individually
    # still exceed eps.
    candidate_norm = candidate_sq.sqrt().clamp_min(1e-8)
    gradient_norm = gradient_sq.sqrt().clamp_min(1e-8)
    denominator = candidate_norm * gradient_norm

    return (numerator / denominator).clamp(-1.0, 1.0), gradient_sq > 0.0


class MAL_SGD(Optimizer):
    r"""Memory-ALigned heavy-ball SGD.

    Let :math:`m_t` be the existing momentum buffer and :math:`g_t` the current (possibly
    L2-regularized) gradient. MAL first constructs the direction that the corresponding
    base optimizer would apply using the fixed coefficient :math:`\beta`. For heavy-ball
    SGD this is

    :math:`\hat{u}_t = \hat{m}_t = \beta m_t + g_t`,

    while PyTorch-style Nesterov SGD uses

    :math:`\hat{u}_t = g_t + \beta\hat{m}_t`.

    MAL measures :math:`s_t=\cos(\hat{u}_t,g_t)` and commits
    :math:`m_{t+1} \gets c_t \, m_t + g_t`, where
    :math:`c_t=(1+s_t)/2` for a non-zero gradient. Thus ``beta`` controls the fixed
    candidate probe while the smoothly adaptive :math:`c_t` controls the update.

    Under Nesterov, the applied direction is :math:`g_t + c_t \, m_t` using the *updated*
    buffer (PyTorch/Sutskever form). MAL is deliberately a time-varying-coefficient
    variant of that rule.

    A zero buffer makes the probe self-aligned (:math:`c = 1`), so the first step is
    undamped; a zero gradient carries no alignment evidence, so it falls back to the
    fixed ``beta`` rather than reading as an artificial 0.5.

    Alignment is always measured over the entire parameter tensor, producing one scalar
    effective coefficient per parameter tensor and step.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        pwr: float = 1.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < pwr <= 1.0:
            raise ValueError(f"Invalid p value: {pwr}")
        if nesterov and beta <= 0.0:
            raise ValueError("Nesterov momentum requires a positive initial beta")

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for p in params:
            if not p.requires_grad:
                continue
            # Exclude biases and 1D normalization parameters from weight decay
            if weight_decay == 0.0 or p.ndim <= 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        if not decay_params and not no_decay_params:
            raise ValueError("Optimizer received no trainable parameters.")

        optim_groups = []

        if no_decay_params:
            optim_groups.append(
                {
                    "params": no_decay_params,
                    "momentum": [torch.zeros_like(p) for p in no_decay_params],
                    "weight_decay": 0.0,
                }
            )
        if decay_params:
            optim_groups.append(
                {
                    "params": decay_params,
                    "momentum": [torch.zeros_like(p) for p in decay_params],
                    "weight_decay": weight_decay,
                },
            )

        defaults = {
            "lr": lr,
            "beta": beta,
            "pwr": pwr,
            "nesterov": nesterov,
        }  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            pwr = group["pwr"]
            nesterov = group["nesterov"]
            for p, m in zip(group["params"], group["momentum"]):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    # Coupled weight decay without mutating "p.grad" in-place.
                    g = g.add(p, alpha=wd)

                # Probe the direction the base optimizer would apply before MAL edits
                # its memory coefficient. Cosine is invariant to positive scaling, so
                # the Nesterov direction
                #   g + beta*(beta*m + g) = beta**2*m + (1 + beta)*g
                # can be normalized to g + beta**2/(1 + beta)*m and evaluated by the
                # allocation-free affine expansion in ``_get_cosine_sim``.
                probe_momentum_scale = beta
                if nesterov:
                    probe_momentum_scale = beta**2 / (1.0 + beta)

                cosine_sim, has_gradient = _get_cosine_sim(
                    m,
                    g,
                    probe_momentum_scale,
                )
                retention = ((1.0 + cosine_sim) * 0.5) ** pwr

                # Effective momentum coefficient for this step
                eff_beta = torch.where(has_gradient, retention, beta)

                m.mul_(eff_beta).add_(g)

                if nesterov:
                    # PyTorch/Sutskever form with the same current coefficient:
                    p.sub_(g, alpha=lr)
                    p.addcmul_(m, eff_beta, value=-lr)
                else:
                    p.sub_(m, alpha=lr)

        return loss


class MAL_ADAMW(Optimizer):
    r"""Memory-ALigned AdamW for transformer training (ViT / LLM).

    The alignment gate modulates only the first moment's EMA coefficient, with one
    scalar coefficient per parameter tensor. Crucially, MAL scores the proposed
    *parameter direction*, not the unpreconditioned first moment. Given probe moments

    :math:`m_t^{probe}=\beta_1m_{t-1}+(1-\beta_1)g_t`,

    :math:`v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2`,

    it forms the AdamW direction

    :math:`u_t^{probe}=\hat m_t^{probe}/(\sqrt{\hat v_t}+\epsilon)`

    and measures :math:`s_t=\cos(u_t^{probe},g_t)`. This matters because Adam's
    positive diagonal preconditioner preserves coordinate signs but generally changes
    a whole-tensor cosine.

    The candidate always uses the fixed ``beta1``. MAL then applies the smooth adaptive
    coefficient :math:`c_t=(1+s_t)/2` on every aligned parameter step.

    The committed EMA is m <- c*m + (1-c)*g. The second moment (scale tracker,
    beta2) and its bias correction are standard AdamW and are never gated.

    First-moment bias correction is exact under dynamic per-tensor c: the running
    product ``beta_product = prod_s c_s`` is tracked per parameter, and the correction is
    ``1 - beta_product`` (reduces to ``1 - beta1**t`` for constant c). Smooth c is capped at
    1 - 1e-4 because c = 1 is the one degenerate EMA value (zero mass on g freezes
    the memory and zeroes the correction). A perfectly aligned first adaptive step
    has the same bias-corrected direction as AdamW, but deliberately a much longer
    raw first-moment horizon.

    Weight decay is decoupled (AdamW) and never enters the alignment signal.
    LayerNorm gains and biases (ndim <= 1) keep a fixed beta1 unless align_1d=True:
    cosine similarity on small tensors can be noise-dominated.

    Second-moment step counts are per parameter and serialized, so intermittent
    gradients and checkpoint resume retain the exact AdamW bias correction.
    """

    MAX_BETA1 = 1.0 - 1e-4

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        pwr: float = 1.0,
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
                        "step": [0 for _ in group_params],
                        "beta_product": [torch.tensor(1.0, device=p.device) for p in group_params],
                        "align_1d": align_1d,
                    }
                )

        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "pwr": pwr,
            "eps": eps,
            "align_1d": align_1d,
        }
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
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            pwr = group["pwr"]
            eps = group["eps"]
            steps = group["step"]
            beta_products = group["beta_product"]
            align_1d = group["align_1d"]

            for i, (p, m, v, beta_product) in enumerate(zip(group["params"], group["m"], group["v"], beta_products)):
                if p.grad is None:
                    continue

                g = p.grad
                steps[i] += 1
                bc2_sqrt = (1.0 - beta2 ** steps[i]) ** 0.5

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay; never enters the gate

                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                denominator = v.sqrt().div_(bc2_sqrt).add_(eps)

                if p.ndim > 1 or align_1d:
                    # Alignment of the proposed, fully preconditioned AdamW update
                    # with the fresh gradient. First-moment bias correction is a
                    # positive tensor-wide scalar, so it cannot change this cosine.
                    m_probe = m.lerp(g, 1.0 - beta1)
                    u_probe = m_probe.div(denominator)
                    cosine_sim, has_gradient = _get_cosine_sim(u_probe, g)
                    retention = (1.0 + cosine_sim) * 0.5

                    # Keep the coefficient in stable scalar-state precision. In fp16
                    # or bf16, MAX_BETA1 would otherwise round to 1 and zero the bias
                    # correction on strongly aligned steps.
                    retention = (retention.to(beta_product.dtype)) ** pwr
                    eff_beta1 = torch.where(has_gradient, retention, beta1).clamp_max(self.MAX_BETA1)

                else:
                    eff_beta1 = beta1  # tiny 1-D params: plain EMA

                m.lerp_(g, 1.0 - eff_beta1)  # m <- c*m + (1-c)*g

                # AdamW update with exact first-moment bias correction under dynamic c
                beta_product.mul_(eff_beta1)
                u = m.div(1.0 - beta_product).div_(denominator)
                p.add_(u, alpha=-lr)

        return loss
