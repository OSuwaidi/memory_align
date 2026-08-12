from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer

MAL_MODES: tuple[str, str] = ("smooth", "conflict_only")


def _get_cosine_sim(
    momentum_or_candidate: torch.Tensor,
    gradient: torch.Tensor,
    momentum_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate/gradient cosine similarity and gradient non-zero status.

    When ``momentum_scale`` is omitted, ``momentum_or_candidate`` is the already
    materialized candidate. When it is supplied, the candidate is
    ``momentum_scale * momentum_or_candidate + gradient``. Whole-tensor
    alignment expands that affine candidate into three vector dot products so
    it never needs to be materialized.

    A zero fresh gradient has no direction, so callers use ``has_gradient`` to retain their prior beta coefficient.

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
    denominator = (candidate_sq * gradient_sq).sqrt().clamp_min(1e-8)

    return (numerator / denominator).clamp(-1.0, 1.0), gradient_sq > 0.0


class MAL_SGD(Optimizer):
    r"""Memory-ALigned heavy-ball SGD with two explicit retention rules.

    Let :math:`m_t` be the existing momentum buffer and :math:`g_t` the current (possibly
    L2-regularized) gradient. All modes probe a momentum candidate

    :math:`\hat{m_t} = \beta_{probe} \, m_t + g_t`

    and measure :math:`s_t=\cos(\hat{m_t},g_t)`. The committed state is always
    :math:`m_{t+1} \gets c_t \, m_t + g_t`, with ``mode`` selecting :math:`c_t`:

    * ``"smooth"`` is the current fully adaptive MAL rule. It uses the previous
      coefficient as :math:`\beta_{probe}` and sets
      :math:`c_t=(1+s_t)/2` at every non-zero-gradient step. The result is stored as
      the next per-parameter probe coefficient. This mode does not use ''beta''.
    * ``"conflict_only"`` keeps the same fixed ``beta`` whenever
      :math:`s_t\geq0`. On conflict, it applies the smooth rule as an additional
      multiplier: :math:`c_t=\beta(1+s_t)/2`. This is the literal fixed-beta-plus-
      conflict-decay ablation; it intentionally has a threshold discontinuity.

    Under Nesterov, the applied direction is :math:`g_t + c_t \, m_t` using the *updated*
    buffer (PyTorch/Sutskever form), so a fixed-mode step with ``c == beta`` coincides with
    ``torch.optim.SGD(momentum=beta, nesterov=True)``. MAL is deliberately a
    time-varying-coefficient variant of that rule.

    The smooth mode has no momentum hyperparameter. Its coefficient is recomputed from
    alignment every step, and the recursion has a closed-form operating point: under
    isotropic gradient noise :math:`\langle m, g \rangle \to 0`, so the steady state
    :math:`\|m\|^2 = c^2 \|m\|^2 + \|g\|^2` gives
    :math:`\cos(\hat{m}, g) = \sqrt{1 - c^2}` and the fixed point solves
    :math:`2c - 1 = \sqrt{1 - c^2}`, i.e. :math:`5c^2 = 4c`, so

    :math:`c^{*} = 4/5`

    independent of dimension — an effective memory-kernel mass of 5 and horizon of 4 steps.
    Deviations above 4/5 measure per-tensor gradient signal-to-noise: driving the recursion
    with :math:`g = s + \xi` gives :math:`c^{*} = 0.80, 0.88, 0.95` at SNR
    :math:`\|s\|/\|\xi\| = 0, 1, 2`. See ``diagnostics/fixed_point.py`` for the verification.

    A zero buffer makes the probe self-aligned (:math:`c = 1`), so the first step is
    undamped; a zero gradient carries no alignment evidence, and the previous coefficient
    is retained rather than reading as an artificial 0.5.

    Alignment is always measured over the entire parameter tensor, and each parameter
    tensor stores one scalar coefficient.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        mode: str = "smooth",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if mode not in MAL_MODES:
            raise ValueError(f"Invalid MAL mode: {mode!r}; expected one of {MAL_MODES}")
        if nesterov and beta <= 0.0:
            raise ValueError("Nesterov momentum requires a positive beta in fixed-beta MAL modes")

        self.mode = mode

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

        optim_groups = []

        if no_decay_params:
            optim_groups.append(
                {
                    "params": no_decay_params,
                    "momentum": no_decay_momentum,
                    "weight_decay": 0.0,
                    "beta": beta,
                }
            )
        if decay_params:
            optim_groups.append(
                {
                    "params": decay_params,
                    "momentum": decay_momentum,
                    "weight_decay": weight_decay,
                    "beta": beta,
                },
            )

        defaults = {
            "lr": lr,
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
            wd = group["weight_decay"]
            beta = group["beta"]
            nesterov = group["nesterov"]

            for i, (p, m) in enumerate(zip(group["params"], group["momentum"])):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    # Coupled weight decay without mutating "p.grad" in-place.
                    g = g.add(p, alpha=wd)

                # Probe the ordinary momentum candidate before committing the MAL edit.
                cosine_sim, has_gradient = _get_cosine_sim(
                    m,
                    g,
                    beta,
                )
                d = (1.0 - cosine_sim) * 0.5  # normalized cosine distance
                retention = 1.0 - d

                # Effective momentum coefficient for this step
                if self.mode == "smooth":
                    eff_beta = torch.where(has_gradient, retention, beta)
                else:  # conflict_only
                    conflict = has_gradient & cosine_sim.lt(0.0)
                    eff_beta = torch.where(conflict, beta * retention, beta)

                m.mul_(eff_beta).add_(g)

                if nesterov:
                    # PyTorch/Sutskever form with the same current coefficient:
                    p.sub_(g, alpha=lr)
                    p.addcmul_(m, eff_beta, value=-lr)
                else:
                    p.sub_(m, alpha=lr)

        return loss


class MAL_ADAMW(Optimizer):
    """Memory-ALigned AdamW for transformer training (ViT / LLM).
    The alignment gate modulates ONLY the first moment's EMA coefficient, per
    parameter tensor.
    Probe: the candidate EMA m_hat = beta_probe*m + (1-beta_probe)*g vs. the fresh
    gradient. ``mode="smooth"`` uses the preceding coefficient as ``beta_probe``
    and sets c = 1-d at every step. ``mode="conflict_only"`` probes with the fixed
    beta1 and keeps c = beta1 unless the candidate conflicts with the gradient, in
    which case c = beta1*(1-d).
    The committed EMA is m <- c*m + (1-c)*g. The second moment (scale tracker,
    beta2) and its bias correction are standard AdamW and are never gated.

    First-moment bias correction is exact under dynamic per-tensor c: the running
    product bc_prod = prod_s c_s is tracked per parameter, and the correction is
    1 - bc_prod (reduces to 1 - beta1^t for constant c). Smooth c is capped at
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
        mode: str = "smooth",
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
        if mode not in MAL_MODES:
            raise ValueError(f"Invalid MAL mode: {mode!r}; expected one of {MAL_MODES}")

        beta1 = betas[0]
        self.mode = mode
        self.base_beta1 = beta1
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
                        "base_beta1": beta1,
                        "mal_mode": mode,
                        "align_1d": align_1d,
                    }
                )

        defaults = {
            "lr": lr,
            "beta2": betas[1],
            "eps": eps,
            "base_beta1": beta1,
            "mal_mode": mode,
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
            beta2 = group["beta2"]
            eps = group["eps"]
            betas = group["beta"]
            bc_prods = group["bc_prod"]
            steps = group["step"]
            base_beta1 = float(group.get("base_beta1", self.base_beta1))
            mode = group.get("mal_mode", self.mode)
            align_1d = bool(group.get("align_1d", self.align_1d))

            for i, (p, m, v) in enumerate(zip(group["params"], group["m"], group["v"])):
                if p.grad is None:
                    continue

                g = p.grad
                steps[i] += 1
                bc2_sqrt = (1.0 - beta2 ** steps[i]) ** 0.5

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay; never enters the gate

                beta1_state = betas[i]
                if p.ndim > 1 or align_1d:
                    # Alignment of the candidate EMA with the fresh gradient:
                    beta_probe: float | torch.Tensor = beta1_state if mode == "smooth" else base_beta1
                    m_hat = torch.lerp(g, m, beta_probe)
                    cosine_sim, has_gradient = _get_cosine_sim(m_hat, g)
                    d = (1.0 - cosine_sim) * 0.5  # normalized cosine distance
                    retention = 1.0 - d

                    if mode == "smooth":
                        proposed_c = retention.clamp_max(self.MAX_BETA1)
                        c = torch.where(has_gradient, proposed_c, beta1_state)
                        c = beta1_state.copy_(c)
                    else:  # conflict_only
                        conflict = has_gradient & cosine_sim.lt(0.0)
                        c = beta1_state.copy_(retention).mul_(base_beta1).masked_fill_(~conflict, base_beta1)
                else:
                    c = beta1_state.fill_(base_beta1)  # tiny 1-D params: plain EMA

                m.lerp_(g, 1.0 - c)  # m <- c*m + (1-c)*g
                bc_prods[i].mul_(c)
                bc_prod = bc_prods[i]
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # AdamW update with exact first-moment bias correction under dynamic c
                u = m.div(v.sqrt().div_(bc2_sqrt).add_(eps))
                u.div_(1.0 - bc_prod)
                p.add_(u, alpha=-lr)

        return loss
