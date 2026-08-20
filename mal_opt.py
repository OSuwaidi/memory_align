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

    A zero-fresh gradient has no direction, so callers use ``has_gradient`` to select their configured fallback coefficient.

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


class MAL_SGDM(Optimizer):
    r"""Memory-ALigned heavy-ball SGD.

    Let :math:`m_{t-1}` be the stored momentum buffer and :math:`g_t` the current
    (possibly L2-regularized) gradient. MAL first probes the direction that the
    corresponding base optimizer would apply using the fixed coefficient
    :math:`\beta`. The proposed plain heavy-ball buffer is

    :math:`\hat{m}_t = \beta m_{t-1} + g_t`.

    Thus, the heavy-ball probe is :math:`\hat{u}_t=\hat{m}_t`, while the
    PyTorch-style Nesterov probe is

    :math:`\hat{u}_t = g_t + \beta\hat{m}_t`.

    MAL measures :math:`s_t=\cos(\hat{u}_t,g_t)` and computes
    :math:`c_t=((1+s_t)/2)^{\mathrm{pwr}}` for a non-zero gradient. Heavy-ball
    applies :math:`u_t=g_t+c_t m_{t-1}`. Nesterov applies
    :math:`u_t=g_t+c_t m_t`, where :math:`m_t` is the buffer selected below
    (PyTorch/Sutskever form).

    With ``in_place=False``, the original MAL formulation is used: the stored
    buffer advances independently as :math:`m_t=\beta m_{t-1}+g_t`, so
    :math:`c_t` reweights only the direction applied on the current step. With
    ``in_place=True``, the adaptive coefficient is also written into memory:
    :math:`m_t=c_t m_{t-1}+g_t`. In either mode, the alignment probe itself uses
    the fixed ``beta`` so variants are scored against the same base optimizer.

    With ``scale=True``, the final applied direction is rescaled to the norm of
    the corresponding fixed-beta probe. This preserves the base optimizer's
    step magnitude while retaining MAL's change in direction.

    A zero buffer makes the probe self-aligned (:math:`c_t=1`). A zero gradient
    carries no alignment evidence, so :math:`c_t` falls back to the fixed ``beta``
    rather than treating an undefined direction as an artificial 0.5.

    **Granularity.** With ``per_unit=False`` alignment is measured over the whole
    parameter tensor, giving one scalar :math:`c_t` per tensor and step. With
    ``per_unit=True`` it is measured per *output unit*: the cosine reduces over the
    fan-in axes, so a ``Linear`` gets one coefficient per neuron (reduce ``dim=1``)
    and a ``ConvNd`` one per output kernel (reduce ``dims=1..N-1``). The resulting
    coefficient has shape ``(out, 1, ..., 1)`` and broadcasts against the update, and
    ``scale`` likewise matches each unit's step norm to that unit's fixed-beta probe
    norm. Parameters with ``ndim <= 1`` (biases, norm affines) always use the
    whole-tensor cosine: their "units" are single scalars, whose cosine could only be
    :math:`\pm 1`.

    Finer granularity lets the gate rotate the applied update further away from the
    plain heavy-ball direction -- it is a per-layer scalar knob at ``per_unit=False``
    and a per-neuron one at ``per_unit=True`` -- at the cost of a noisier cosine on
    units with small fan-in (a stem conv with fan-in 27 is the extreme case).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        pwr: float = 1.0,
        in_place: bool = False,
        scale: bool = False,
        nesterov: bool = False,
        per_unit: bool = False,
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

        for group_params, group_wd in ((no_decay_params, 0.0), (decay_params, weight_decay)):
            if group_params:
                optim_groups.append(
                    {
                        "params": group_params,
                        "momentum": [torch.zeros_like(p) for p in group_params],
                        "weight_decay": group_wd,
                    }
                )

        defaults = {
            "lr": lr,
            "beta": beta,
            "pwr": pwr,
            "in_place": in_place,
            "scale": scale,
            "nesterov": nesterov,
            "per_unit": per_unit,
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
            in_place = group["in_place"]
            scale = group["scale"]
            nesterov = group["nesterov"]
            per_unit = group["per_unit"]

            for p, m in zip(group["params"], group["momentum"]):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    # Coupled weight decay without mutating "p.grad" in-place.
                    g = g.add(p, alpha=wd)

                probe_m = torch.add(g, m, alpha=beta)
                probe_u = torch.add(g, probe_m, alpha=beta) if nesterov else probe_m

                # Fan-in axes for per-output-unit alignment; None keeps the
                # whole-tensor cosine (also the only sane choice for ndim <= 1,
                # where each "unit" is a lone scalar).
                dims = tuple(range(1, p.ndim)) if (per_unit and p.ndim > 1) else None

                if dims is None:
                    g_norm = torch.linalg.vector_norm(g)
                    probe_norm = torch.linalg.vector_norm(probe_u)
                    dot = torch.vdot(g.flatten(), probe_u.flatten())
                else:
                    # vector_norm, not Tensor.norm: the latter dispatches to
                    # matrix_norm for a tuple dim and raises on 3+ reduced axes.
                    g_norm = torch.linalg.vector_norm(g, dim=dims, keepdim=True)
                    probe_norm = torch.linalg.vector_norm(probe_u, dim=dims, keepdim=True)
                    dot = (g * probe_u).sum(dim=dims, keepdim=True)

                denominator = g_norm.clamp_min(1e-8) * probe_norm.clamp_min(1e-8)
                cosine_sim = (dot / denominator).clamp(-1.0, 1.0)

                eff_beta = ((1.0 + cosine_sim) * 0.5) ** pwr  # effective momentum coefficient for this step
                eff_beta = torch.where(g_norm > 0.0, eff_beta, beta)
                eff_m = torch.addcmul(g, m, eff_beta)  # eff_beta * m_{t-1} + g

                if in_place:
                    m.copy_(eff_m)
                else:
                    m.copy_(probe_m)  # plain heavy ball

                applied = torch.addcmul(g, m, eff_beta) if nesterov else eff_m

                if scale:
                    applied_norm = (
                        torch.linalg.vector_norm(applied)
                        if dims is None
                        else torch.linalg.vector_norm(applied, dim=dims, keepdim=True)
                    )
                    applied.mul_(probe_norm / applied_norm.clamp_min(1e-8))

                p.sub_(applied, alpha=lr)

        return loss


class MAL_AdamW(Optimizer):
    r"""Memory-ALigned AdamW for transformer training (ViT / LLM).

    The alignment gate modulates only the first moment applied on the current step,
    with one scalar coefficient per parameter tensor; it does not alter the stored
    EMA. Crucially, MAL scores the proposed
    *parameter direction*, not the (unpreconditioned) first moment. Given probe moments

    :math:`m_t^{probe}=\beta_1m_{t-1}+(1-\beta_1)g_t`,

    :math:`v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2`,

    it forms the AdamW direction

    :math:`u_t^{probe}=\hat m_t^{probe}/(\sqrt{\hat v_t}+\epsilon)`

    and measures :math:`s_t=\cos(u_t^{probe},g_t)`. This matters because Adam's
    positive diagonal preconditioner preserves coordinate signs but generally changes
    a whole-tensor cosine.

    The candidate always uses the fixed ``beta1``. On every aligned parameter step,
    MAL applies the smooth adaptive coefficient
    :math:`c_t=((1+s_t)/2)^{\mathrm{pwr}}`, capped below one for numerical stability.

    **The stored buffer is ungated.** The applied first moment is the one-step
    reweighting ``c*m_{t-1} + (1-c)*g``, while the stored EMA advances independently
    as plain AdamW, ``m <- beta1*m + (1-beta1)*g``, and never sees ``c``. The second
    moment (scale tracker, beta2) and its bias correction are standard AdamW and are
    never gated.

    Because the buffer stays a plain EMA, first-moment bias correction is exact in closed
    form: ``E[m_{t-1}] = (1 - beta1**(t-1))*g`` gives
    ``E[c*m_{t-1} + (1-c)*g] = (1 - c*beta1**(t-1))*g``, so the correction is
    ``1 - c*beta1**(t-1)`` -- no running product of past coefficients, hence no accumulated
    rounding drift, and it reduces to ``1 - beta1**t`` whenever ``c == beta1``. Smooth c is
    capped at 1 - 1e-3 because c = 1 is the one degenerate EMA value (zero mass on g).

    Weight decay is decoupled (AdamW) and never enters the alignment signal.
    Alignment is measured for every trainable parameter tensor, including biases
    and normalization affines. Parameters with ``ndim <= 1`` are excluded only
    from weight decay.

    Second-moment step counts are per parameter and serialized, so intermittent
    gradients and checkpoint resume retain the exact AdamW bias correction.
    """

    MAX_BETA1 = 1.0 - 1e-3

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        pwr: float = 1.0,
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
        if not 0.0 < pwr <= 1.0:
            raise ValueError(f"Invalid p value: {pwr}")

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

        for group_params, group_wd in ((no_decay_params, 0.0), (decay_params, weight_decay)):
            if group_params:
                optim_groups.append(
                    {
                        "params": group_params,
                        "m": [torch.zeros_like(p) for p in group_params],
                        "v": [torch.zeros_like(p) for p in group_params],
                        "weight_decay": group_wd,
                        "step": [0 for _ in group_params],
                    }
                )

        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "pwr": pwr,
            "eps": eps,
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

            for i, (p, m, v) in enumerate(zip(group["params"], group["m"], group["v"])):
                if p.grad is None:
                    continue

                g = p.grad
                steps[i] += 1
                bc2_sqrt = (1.0 - beta2 ** steps[i]) ** 0.5

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay; never enters the gate

                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                denominator = v.sqrt().div_(bc2_sqrt).add_(eps)

                # Alignment of the proposed, fully preconditioned AdamW update
                # with the fresh gradient. First-moment bias correction is a
                # positive tensor-wide scalar, so it cannot change this cosine.
                u_probe = m.lerp(g, 1.0 - beta1).div(denominator)
                cosine_sim, has_gradient = _get_cosine_sim(u_probe, g)
                retention = (1.0 + cosine_sim) * 0.5

                # Keep low-precision coefficients in stable scalar precision. In
                # fp16 or bf16, MAX_BETA1 would otherwise round to 1 and zero the
                # bias correction on strongly aligned steps. Preserve float64 when
                # the optimizer is explicitly used with float64 parameters.
                if retention.dtype in (torch.float16, torch.bfloat16):
                    retention = retention.to(torch.float32)
                retention = retention**pwr
                retention = retention.clamp_max(self.MAX_BETA1)
                eff_beta1 = torch.where(has_gradient, retention, beta1)

                # Ungated stored memory: the coefficient reweights only this step's applied
                # direction, while the buffer advances independently at fixed beta1.
                eff_m = torch.lerp(g, m, eff_beta1)  # c*m_{t-1} + (1-c)*g
                m.lerp_(g, 1.0 - beta1)  # plain AdamW EMA, untouched by MAL

                # Bias correction, exact and closed form. Because the buffer is a plain EMA,
                #   E[m_{t-1}] = (1 - beta1**(t-1)) * g,
                # so the applied reweighting has
                #   E[c*m_{t-1} + (1-c)*g] = (1 - c*beta1**(t-1)) * g.
                # No running product of past coefficients is needed -- which also means the
                # correction cannot drift with accumulated rounding and reduces to the
                # familiar 1 - beta1**t whenever c == beta1.
                correction = 1.0 - eff_beta1 * (beta1 ** (steps[i] - 1))
                u = eff_m.div_(correction).div_(denominator)
                p.add_(u, alpha=-lr)

        return loss
