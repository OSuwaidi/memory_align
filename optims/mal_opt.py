from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer


def get_norms_and_eff_beta(
    g: torch.Tensor,
    probe: torch.Tensor,
    pwr: float,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g_norm = torch.linalg.vector_norm(g)
    probe_norm = torch.linalg.vector_norm(probe)
    dot = torch.dot(g.flatten(), probe.flatten())

    denominator = (g_norm * probe_norm) + eps
    cosine_sim = (dot / denominator).clamp(-1.0, 1.0)

    return g_norm, probe_norm, ((1.0 + cosine_sim) * 0.5) ** pwr


def _apply_gate(base_beta: float, gate: torch.Tensor, gate_mode: str) -> torch.Tensor:
    """Map an alignment gate in [0, 1] to the memory coefficient."""
    if gate_mode == "attenuate":
        return gate.mul(base_beta)
    if gate_mode == "replace":
        return gate
    raise ValueError(f"Invalid gate_mode value: {gate_mode}")


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

    MAL measures :math:`s_t=\cos(\hat{u}_t,g_t)` and computes the alignment
    gate :math:`q_t=((1+s_t)/2)^{\mathrm{pwr}}` for a non-zero gradient. With
    ``gate_mode="replace"`` (the historical implementation), the applied memory
    coefficient is :math:`c_t=q_t`. With ``gate_mode="attenuate"``, it is
    :math:`c_t=\beta q_t`; this is a literal gate on the base optimizer.
    Attenuation keeps :math:`c_t\in[0,\beta]`. Heavy-ball applies
    :math:`u_t=g_t+c_t m_{t-1}`. Nesterov applies :math:`u_t=g_t+c_t m_t`, where
    :math:`m_t` is the buffer selected below (PyTorch/Sutskever form).

    With ``in_place=False``, the original MAL formulation is used: the stored
    buffer advances independently as :math:`m_t=\beta m_{t-1}+g_t`, so
    :math:`c_t` reweights only the direction applied on the current step. With
    ``in_place=True``, the adaptive coefficient is also written into memory:
    :math:`m_t=c_t m_{t-1}+g_t`. In either mode, the alignment probe itself uses
    the fixed ``beta`` so variants are scored against the same base optimizer.
    Replacement plus in-place state has no uniform contraction because
    :math:`c_t` can equal one; attenuation keeps :math:`c_t\leq\beta<1`.

    With ``scale=True``, the final applied direction is rescaled to the norm of
    the corresponding fixed-beta probe. This preserves the base optimizer's
    step magnitude while retaining MAL's change in direction.

    A zero buffer makes the probe self-aligned (:math:`q_t=1`). A zero gradient
    carries no alignment evidence, so :math:`c_t` falls back to the fixed ``beta``
    rather than treating an undefined direction as an artificial 0.5.

    Alignment is measured once over each complete parameter tensor, producing one
    scalar gate per tensor and optimizer step.
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
        gate_mode: str = "attenuate",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if pwr not in (0.5, 1.0):
            raise ValueError(f"Invalid p value: {pwr}")
        if nesterov and beta <= 0.0:
            raise ValueError("Nesterov momentum requires a positive initial beta")
        if gate_mode not in ("replace", "attenuate"):
            raise ValueError(f"Invalid gate_mode value: {gate_mode}")

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
            "gate_mode": gate_mode,
        }  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

    @torch.no_grad()
    def step(self, closure: Callable[[], float | torch.Tensor] | None = None) -> Any:
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
            gate_mode = group["gate_mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]  # used such that loading model form checkpoint pushes all its weights + states to correct device
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                m = state["momentum_buffer"]
                if wd > 0.0:
                    # Coupled weight decay
                    g = g.add(p, alpha=wd)

                m_probe = g.add(m, alpha=beta)
                u = g.add(m_probe, alpha=beta) if nesterov else m_probe

                g_norm, u_norm, beta_eff = get_norms_and_eff_beta(g, u, pwr)
                beta_eff = _apply_gate(beta, beta_eff, gate_mode)
                beta_eff = torch.where(g_norm > 0.0, beta_eff, beta)
                m_eff = torch.addcmul(g, m, beta_eff)  # beta_eff * m_{t-1} + g

                if in_place:
                    m.copy_(m_eff)
                else:
                    m.copy_(m_probe)  # plain heavy ball

                u_eff = torch.addcmul(g, m, beta_eff) if nesterov else m_eff

                if scale:
                    u_eff_norm = torch.linalg.vector_norm(u_eff) + 1e-7
                    u_eff.mul_(u_norm / u_eff_norm)

                p.sub_(u_eff, alpha=lr)

        return loss


class MAL_AdamW(Optimizer):
    r"""Memory-ALigned AdamW.

    Let :math:`m_{t-1}, v_{t-1}` be the stored first/second moments and :math:`g_t`
    the current gradient. MAL first probes the step the base AdamW would take with
    the fixed coefficient :math:`\beta_1`:

    :math:`\hat{m}_t = \beta_1 m_{t-1} + (1-\beta_1) g_t`,\
    :math:`D_t = \sqrt{\hat{v}_t} + \epsilon`,\
    :math:`\hat{u}_t = \hat{m}_t / (r^{probe}_t D_t)`,

    where :math:`v_t` always advances with the fixed :math:`\beta_2` (MAL gates the
    *direction* memory only) and :math:`r^{probe}_t` is the coefficient
    normalization below. The alignment cosine :math:`s_t` is measured according
    to ``align``:

    - ``"update"``: :math:`\cos(g_t,D_t^{-1}\hat m_t)`, the direct Euclidean
      angle between the local gradient and AdamW's applied probe. Its numerator
      is the first-order descent term :math:`g_t^T D_t^{-1}\hat m_t`.
    - ``"metric"``: :math:`\cos(D_t^{-1/2}g_t,D_t^{-1/2}\hat m_t)`. Its numerator
      is proportional to the preconditioned descent term
      :math:`g_t^T D_t^{-1}\hat m_t`, making this the cleanest geometry for theory.
    - ``"white"``: :math:`\cos(D_t^{-1}g_t,D_t^{-1}\hat m_t)`, the angle after
      transforming both vectors by AdamW's diagonal map. Unlike ``"update"``
      and ``"metric"``, its numerator need not have the sign of the local
      descent term, so it is retained as an empirical ablation rather than the
      theorem-facing geometry.
    - ``"moment"``: :math:`\cos(g_t,\hat m_t)`, the raw first-moment geometry and
      the most literal extension of MAL-SGDM.

    MAL computes :math:`q_t=((1+s_t)/2)^{\mathrm{pwr}}`. ``gate_mode`` maps this
    to the applied memory coefficient :math:`c_t`:

    - ``"replace"`` keeps the historical rule :math:`c_t=q_t`, which can either
      attenuate or amplify memory relative to :math:`\beta_1`.
    - ``"attenuate"`` uses :math:`c_t=\beta_1q_t`, so
      :math:`c_t\in[0,\beta_1]`. Then the effective moment
      :math:`\tilde m_t=c_tm_{t-1}+(1-\beta_1)g_t` lies on the segment from the
      memoryless raw moment to the AdamW probe, and perfect alignment recovers
      AdamW exactly. This is the literal gating interpretation.
    ``gradient_weight_mode`` controls the coefficient of the fresh gradient.
    Define :math:`a_t` as the actual memory coefficient above. ``"fixed"``
    preserves the historical MAL rule

    :math:`\tilde m_t=a_tm_{t-1}+(1-\beta_1)g_t`.

    ``"complement"`` instead uses

    :math:`\tilde m_t=a_tm_{t-1}+(1-a_t)g_t`.

    The latter is a convex adaptive EMA and implements the proposed rule
    :math:`a_t=\beta_1q_t`, :math:`1-a_t=1-\beta_1q_t`. It is intentionally
    restricted to ``gate_mode="attenuate"``: replacement gives :math:`a_1=1`
    on the self-aligned first step and would therefore produce a zero moment
    and zero normalization factor.

    **Coefficient normalization.** Because :math:`a_t` varies per step, the
    classical :math:`(1-\beta_1^t)` no longer removes zero-initialization
    attenuation from the applied moment. MAL therefore tracks the realized
    scalar coefficient sum :math:`r_t` for each tensor:

    :math:`r^{probe}_t = \beta_1 r_{t-1} + (1-\beta_1)` (equals :math:`1-\beta_1^t`
    when ``in_place=False``) and :math:`r^{eff}_t = a_t r_{t-1} + b_t`,
    where :math:`b_t=1-\beta_1` in ``"fixed"`` mode and
    :math:`b_t=1-a_t` in ``"complement"`` mode,
    and the applied update is :math:`u_t = \tilde{m}_t / (r^{eff}_t D_t)`. With
    the gate frozen at :math:`\beta_1` every correction collapses to
    :math:`1-\beta_1^t` and the update is exactly AdamW, for every ``scale`` mode
    (under the norm-matching modes :math:`r^{eff}` cancels algebraically and only
    :math:`r^{probe}` matters; :math:`r^{eff}` is load-bearing for ``"none"``).

    With ``in_place=False`` (original MAL) the stored buffer advances with the
    fixed :math:`\beta_1` and the gate is transient -- the stored state is then
    *exactly* vanilla-AdamW state. Under attenuation, this is also the
    formulation amenable to a clean bounded-memory analysis because
    :math:`c_t\in[0,\beta_1]`; replacement permits :math:`c_t=1` and memory
    amplification relative to :math:`\beta_1`. ``in_place=True`` writes
    :math:`\tilde{m}_t` (and :math:`r^{eff}_t`) into memory; it was dominated
    everywhere empirically (gate-collapse feedback) and breaks that clean
    decomposition.

    ``scale`` selects where the applied magnitude comes from:

    - ``"step"`` (or ``True``, default): rescale the applied step to the probe
      *step's* norm -- a pure direction correction at AdamW's step length, in the
      whitened geometry. The most overshoot-robust setting measured.
    - ``"moment"``: momentum-space norm matching -- the direction comes from
      :math:`\tilde{m}_t` but the magnitude is inherited from the probe moment,
      :math:`u_t = \tilde{m}_t\,\lVert\hat{m}_t\rVert / (\lVert\tilde{m}_t\rVert\, r^{probe}_t D_t)`.
      Because the magnitude is the probe's, the standard :math:`r^{probe}` is the
      only correction needed (:math:`r^{eff}` cancels algebraically -- it cancels
      under ``"step"`` too; it only matters for ``"none"``). The preconditioner
      then prices the rotated direction into a step. The raw-space counterpart of
      ``"step"``; strongest trust-region profile with ``align="moment"``.
    - ``"none"`` (or ``False``): no matching; misalignment also shrinks the step,
      which the exact correction keeps well-calibrated
      (:math:`u_1 = g_1/D_1` on the first step).

    A zero gradient carries no alignment evidence: :math:`c_t` falls back to
    :math:`\beta_1`. The cosine's numerical floor is pinned at 1e-8 independently
    of ``eps``, matching MAL-SGDM.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-7,
        weight_decay: float = 0.0,
        pwr: float = 1.0,
        align: str = "white",
        in_place: bool = False,
        scale: bool | str = True,
        gate_mode: str = "attenuate",
        gradient_weight_mode="fixed",
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
        if pwr not in (0.5, 1.0):
            raise ValueError(f"Invalid p value: {pwr}")
        if align not in ("update", "metric", "white", "moment"):
            raise ValueError(f"Invalid align value: {align}")
        if isinstance(scale, bool):
            scale = "step" if scale else "none"
        if scale not in ("step", "moment", "none"):
            raise ValueError(f"Invalid scale value: {scale}")
        if gate_mode not in ("replace", "attenuate"):
            raise ValueError(f"Invalid gate_mode value: {gate_mode}")
        if gradient_weight_mode not in ("fixed", "complement"):
            raise ValueError(f"Invalid gradient_weight_mode value: {gradient_weight_mode}")
        if gradient_weight_mode == "complement" and gate_mode != "attenuate":
            raise ValueError('gradient_weight_mode="complement" requires gate_mode="attenuate"')

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
                        "weight_decay": group_wd,
                    }
                )

        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "pwr": pwr,
            "eps": eps,
            "align": align,
            "in_place": in_place,
            "scale": scale,
            "gate_mode": gate_mode,
            "gradient_weight_mode": gradient_weight_mode,
        }
        super().__init__(optim_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float | torch.Tensor] | None = None) -> Any:
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
            align = group["align"]
            in_place = group["in_place"]
            scale = group["scale"]
            gate_mode = group["gate_mode"]
            gradient_weight_mode = group["gradient_weight_mode"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["first_moment_weight"] = torch.zeros((), device=p.device, dtype=p.dtype)

                state["step"] += 1
                step = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                r = state["first_moment_weight"]

                m_probe = m.lerp(g, weight=(1.0 - beta1))
                v.lerp_(g**2, weight=(1.0 - beta2))

                # Equals 1-beta1**t when the stored state follows vanilla AdamW.
                r_probe = beta1 * r + (1.0 - beta1)
                v_unbias = v / (1.0 - beta2**step)
                denominator = v_unbias.sqrt_().add_(eps)

                u = (m_probe / r_probe).div_(denominator)

                if align == "update":
                    grad, dir = g, u
                elif align == "metric":  # cosine in the D^{-1} inner product: numerator is the descent term g^T D^{-1} m, and m=0 is exactly self-aligned
                    d_sqrt = denominator.sqrt()
                    grad, dir = g / d_sqrt, m_probe / d_sqrt
                elif (
                    align == "white"
                ):  # comparing \(D^{-1}g\) with \(D^{-1}m\), whose dot product can have the opposite sign from the actual descent term \(g^\top D^{-1}m\)
                    grad, dir = g / denominator, u
                else:  # align == "moment":
                    grad, dir = g, m_probe

                a_norm, b_norm, beta1_eff = get_norms_and_eff_beta(
                    grad,
                    dir,
                    pwr,
                )
                beta1_eff = _apply_gate(beta1, beta1_eff, gate_mode)
                beta1_eff = torch.where(a_norm > 0.0, beta1_eff, beta1)

                gradient_weight = 1.0 - beta1_eff if gradient_weight_mode == "complement" else 1.0 - beta1
                m_eff = m.mul(beta1_eff).add_(g * gradient_weight)
                r_eff = beta1_eff * r + gradient_weight

                if in_place:
                    m.copy_(m_eff)
                    r.copy_(r_eff)
                else:
                    m.copy_(m_probe)
                    r.copy_(r_probe)

                if scale == "moment":
                    # Raw-space first-moment norm matching: direction from the gated moment, magnitude inherited from
                    # the (unbiased) probe moment. r_eff cancels: u_eff == m_eff * ||m_probe|| / (||m_eff|| * r_probe)
                    m_probe_norm = torch.linalg.vector_norm(m_probe)
                    m_eff_norm = torch.linalg.vector_norm(m_eff) + eps
                    m_eff_unbias = (m_eff / r_probe) * (m_probe_norm / m_eff_norm)
                    u_eff = m_eff_unbias.div_(denominator)

                else:
                    u_eff = (m_eff / r_eff).div_(denominator)
                    if scale == "step":
                        # ||u|| must come from the probe *step*, not the align pair (b is not u under "moment"/"metric")
                        u_probe_norm = b_norm if align in ("update", "white") else torch.linalg.vector_norm(u)
                        u_eff_norm = torch.linalg.vector_norm(u_eff) + eps
                        u_eff.mul_(u_probe_norm / u_eff_norm)

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay

                p.sub_(u_eff, alpha=lr)

        return loss
