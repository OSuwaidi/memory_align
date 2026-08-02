from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer


def _get_cosine_sim(
    candidate: torch.Tensor,
    gradient: torch.Tensor,
    *,
    per_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scale-stable cosine similarity and gradient non-zero status.

    A zero fresh gradient has no direction, so callers use ``has_gradient`` to retain their prior beta coefficient.

    With ``per_output=True``, tensors of two or more dimensions are split along dimension 0:
    Linear weights reduce over ``in_features`` and ConvNd weights over
    ``(in_channels / groups, *kernel_size)``, returning shape ``(out, 1, ..., 1)`` so the
    result broadcasts back over the corresponding weight blocks. One-dimensional
    parameters are always treated as a single vector -- a per-axis cosine there degenerates
    to per-coordinate sign gating, which is the ``hard_per`` failure mode.

    Note the statistical cost: the estimator's noise floor is ``~0.358/sqrt(d)`` in the
    dimension ``d`` being reduced over (see ``diagnostics/fixed_point.py``). Splitting a
    Conv2d weight per output unit drops ``d`` from ``numel`` to the per-unit fan-in, so the
    CIFAR stem (fan-in 27) carries roughly 0.07 of pure estimator noise per unit. The
    stored coefficient acts as the temporal filter that makes this usable.
    """
    if per_output and gradient.ndim > 1:
        reduce_dims = tuple(range(1, gradient.ndim))
        gradient_norm = torch.linalg.vector_norm(gradient, dim=reduce_dims, keepdim=True)
        candidate_norm = torch.linalg.vector_norm(candidate, dim=reduce_dims, keepdim=True)
        denominator = (candidate_norm * gradient_norm).clamp_min(1e-8)
        cosine = ((candidate * gradient).sum(dim=reduce_dims, keepdim=True) / denominator).clamp(-1.0, 1.0)
        return cosine, gradient_norm > 0.0

    candidate_vec = candidate.reshape(-1)
    gradient_vec = gradient.reshape(-1)
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
    L2-regularized) gradient. MAL probes the ordinary momentum candidate

    :math:`\hat{m_t} = \beta_{probe} \, m_t + g_t`

    and maps its cosine with :math:`g_t` to a retention coefficient
    :math:`c_t = (1 + \text{cosine}) / 2 \in [0, 1]`, which is both applied and stored as the
    next probe coefficient (:math:`\beta_{probe} \gets c_t`, per parameter tensor). The
    committed state is :math:`m_{t+1} \gets c_t \, m_t + g_t`.

    Under Nesterov, the applied direction is :math:`g_t + c_t \, m_t` using the *updated*
    buffer (PyTorch/Sutskever form), so a step with ``c == beta`` coincides with
    ``torch.optim.SGD(momentum=beta, nesterov=True)``. MAL is deliberately a
    time-varying-coefficient variant of that rule.

    **There is no momentum hyperparameter.** The coefficient is recomputed from alignment
    every step, and the recursion has a closed-form operating point: under isotropic
    gradient noise :math:`\langle m, g \rangle \to 0`, so the steady state
    :math:`\|m\|^2 = c^2 \|m\|^2 + \|g\|^2` gives
    :math:`\cos(\hat{m}, g) = \sqrt{1 - c^2}` and the fixed point solves
    :math:`2c - 1 = \sqrt{1 - c^2}`, i.e. :math:`5c^2 = 4c`, so

    :math:`c^{*} = 4/5`

    independent of dimension — an effective memory-kernel mass of 5 and horizon of 4 steps.
    Deviations above 4/5 measure per-tensor gradient signal-to-noise: driving the recursion
    with :math:`g = s + \xi` gives :math:`c^{*} = 0.80, 0.88, 0.95` at SNR
    :math:`\|s\|/\|\xi\| = 0, 1, 2`. See ``diagnostics/fixed_point.py`` for the verification.

    A zero buffer makes the probe self-aligned (:math:`c = 1`) so the first step is
    un-damped; a zero gradient carries no alignment evidence and the previous coefficient
    is retained rather than reading as an artificial 0.5.

    ``per_output=True`` measures alignment independently for every output unit of a
    weight tensor (``(out, 1, ..., 1)`` coefficients broadcasting over each unit's block)
    instead of one coefficient per tensor. This is an experimental arm, not the default:
    it produced the best small-CNN numbers on record but did not reproduce on the longer
    protocol, and it trades resolution for estimator variance -- the noise floor rises from
    ``0.358/sqrt(numel)`` to ``0.358/sqrt(fan-in)``. Parameters with ``ndim <= 1`` keep the
    whole-tensor cosine either way.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        per_output: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.per_output = per_output

        decay_params: list[torch.nn.Parameter] = []
        decay_momentum = []
        no_decay_params: list[torch.nn.Parameter] = []
        no_decay_momentum = []

        for p in params:
            if not p.requires_grad:
                continue
            # Exclude biases and 1D normalization parameters from weight decay
            if weight_decay == 0.0 or p.ndim <= 1:
                no_decay_params.append(p)
                no_decay_momentum.append(torch.zeros_like(p))
            else:
                decay_params.append(p)
                decay_momentum.append(torch.zeros_like(p))

        if not decay_params and not no_decay_params:
            raise ValueError("Optimizer received no trainable parameters.")

        def initial_betas(group_params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
            # Zero init: on step 1 the buffer is zero, so the probe reduces to g and the
            # retention is 1 regardless of this value. Under per_output the coefficient is
            # a column vector broadcasting over each output unit's weight block.
            return [p.new_zeros((p.shape[0],) + (1,) * (p.ndim - 1)) if per_output and p.ndim > 1 else p.new_tensor(0.0) for p in group_params]

        optim_groups = []

        if no_decay_params:
            optim_groups.append(
                {
                    "params": no_decay_params,
                    "momentum": no_decay_momentum,
                    "weight_decay": 0.0,
                    "beta": initial_betas(no_decay_params),
                }
            )
        if decay_params:
            optim_groups.append(
                {
                    "params": decay_params,
                    "momentum": decay_momentum,
                    "weight_decay": weight_decay,
                    "beta": initial_betas(decay_params),
                },
            )

        defaults = {
            "lr": lr,
            "nesterov": nesterov,
        }  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

    @torch.no_grad()
    def step(self, closure: Callable[[], float | torch.Tensor] | None = None) -> float | None:
        """Perform a single optimization step."""
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
                beta_probe = group_beta[i]
                m_hat = torch.addcmul(g, m, beta_probe)

                cosine_sim, has_gradient = _get_cosine_sim(m_hat, g, per_output=self.per_output)
                d = (1.0 - cosine_sim) * 0.5  # normalized cosine distance
                retention = 1.0 - d

                # Effective momentum coefficient for this step
                beta = torch.where(has_gradient, retention, beta_probe)
                beta = beta_probe.copy_(beta)

                m.mul_(beta).add_(g)

                if nesterov:
                    # PyTorch/Sutskever form with the same current coefficient:
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
