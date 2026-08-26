from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer


def get_norms_and_eff_beta(
    g: torch.Tensor,
    probe: torch.Tensor,
    pwr: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g_norm = torch.linalg.vector_norm(g)
    probe_norm = torch.linalg.vector_norm(probe).clamp_min(eps)
    dot = torch.dot(g.flatten(), probe.flatten())

    denominator = g_norm.clamp_min(eps) * probe_norm
    cosine_sim = (dot / denominator).clamp(-1.0, 1.0)

    return g_norm, probe_norm, ((1.0 + cosine_sim) * 0.5) ** pwr


def _apply_gate(base_beta: float, gate: torch.Tensor, gate_mode: str) -> torch.Tensor:
    """Map an alignment gate in [0, 1] to the memory coefficient."""
    if gate_mode == "attenuate":
        return gate.mul(base_beta)
    if gate_mode == "cap":
        return gate.clamp_max(base_beta)
    return gate  # historical MAL rule: replace the base coefficient


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
    :math:`c_t=\beta q_t`; this is a literal gate on the base optimizer. The
    exploratory ``gate_mode="cap"`` uses :math:`c_t=\min(\beta,q_t)`, retaining
    more memory while preventing amplification. Both bounded modes keep
    :math:`c_t\in[0,\beta]`. Heavy-ball applies
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
        scale: bool = True,
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
        if gate_mode not in ("replace", "attenuate", "cap"):
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
            "gate_mode": gate_mode,
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
            gate_mode = group["gate_mode"]

            for p, m in zip(group["params"], group["momentum"]):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    # Coupled weight decay
                    g = g.add(p, alpha=wd)

                m_probe = g.add(m, alpha=beta)
                u_probe = g.add(m_probe, alpha=beta) if nesterov else m_probe

                g_norm, u_probe_norm, beta_eff = get_norms_and_eff_beta(g, u_probe, pwr)
                beta_eff = _apply_gate(beta, beta_eff, gate_mode)
                beta_eff = torch.where(g_norm > 0.0, beta_eff, beta)
                m_eff = torch.addcmul(g, m, beta_eff)  # beta_eff * m_{t-1} + g

                if in_place:
                    m.copy_(m_eff)
                else:
                    m.copy_(m_probe)  # plain heavy ball

                u_eff = torch.addcmul(g, m, beta_eff) if nesterov else m_eff

                if scale:
                    u_eff_norm = torch.linalg.vector_norm(u_eff).clamp_min(1e-8)
                    u_eff.mul_(u_probe_norm / u_eff_norm)

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
    *direction* memory only) and :math:`r^{probe}_t` is the exact bias correction
    below. The alignment cosine :math:`s_t` is measured according to ``align``:

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
    - ``"cap"`` is an exploratory bounded replacement,
      :math:`c_t=\min(\beta_1,q_t)`. It also stays in
      :math:`[0,\beta_1]` and recovers AdamW whenever
      :math:`q_t\geq\beta_1`, while damping less aggressively than
      multiplicative attenuation.

    **Exact bias correction.** Because :math:`c_t` varies per step, the classical
    :math:`(1-\beta_1^t)` no longer unbiases the applied moment. Tracking one
    scalar per tensor, :math:`r_t = \mathbb{E}[m_t]/\mathbb{E}[g]`:

    :math:`r^{probe}_t = \beta_1 r_{t-1} + (1-\beta_1)` (equals :math:`1-\beta_1^t`
    when ``in_place=False``) and :math:`r^{eff}_t = c_t r_{t-1} + (1-\beta_1)`,
    and the applied update is :math:`u_t = \tilde{m}_t / (r^{eff}_t D_t)`. With
    the gate frozen at :math:`\beta_1` every correction collapses to
    :math:`1-\beta_1^t` and the update is exactly AdamW, for every ``scale`` mode
    (under the norm-matching modes :math:`r^{eff}` cancels algebraically and only
    :math:`r^{probe}` matters; :math:`r^{eff}` is load-bearing for ``"none"``).

    With ``in_place=False`` (original MAL) the stored buffer advances with the
    fixed :math:`\beta_1` and the gate is transient -- the stored state is then
    *exactly* vanilla-AdamW state, which is also the formulation amenable to
    convergence analysis (bounded :math:`c_t\in[0, 0.999]`, applied direction a
    bounded rotation of the AdamW step toward :math:`g_t`). ``in_place=True``
    writes :math:`\tilde{m}_t` (and :math:`r^{eff}_t`) into memory; it was
    dominated everywhere empirically (gate-collapse feedback) and breaks that
    clean decomposition.

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
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        pwr: float = 1.0,
        align: str = "white",
        in_place: bool = False,
        scale: bool | str = True,
        gate_mode: str = "attenuate",
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
        if gate_mode not in ("replace", "attenuate", "cap"):
            raise ValueError(f"Invalid gate_mode value: {gate_mode}")

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
                        # r = exact E[m]/E[g] correction of the stored buffer (starts at 0: empty memory)
                        "r": [torch.zeros((), device=p.device, dtype=p.dtype) for p in group_params],
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
            "align": align,
            "in_place": in_place,
            "scale": scale,
            "gate_mode": gate_mode,
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
            align = group["align"]
            in_place = group["in_place"]
            scale = group["scale"]
            gate_mode = group["gate_mode"]
            eps = group["eps"]
            steps = group["step"]

            for i, (p, m, v, r) in enumerate(zip(group["params"], group["m"], group["v"], group["r"])):
                if p.grad is None:
                    continue

                g = p.grad
                steps[i] += 1

                m_probe = m.lerp(g, weight=(1.0 - beta1))
                v.lerp_(g**2, weight=(1.0 - beta2))

                r_probe = beta1 * r + (1.0 - beta1)  # exact E[m_probe]/E[g]; equals 1-beta1^t when in_place=False
                v_unbias = v / (1.0 - beta2 ** steps[i])
                denominator = v_unbias.sqrt_().add_(eps)

                u_probe = (m_probe / r_probe).div_(denominator)

                if align == "update":
                    a, b = g, u_probe
                elif align == "metric":  # cosine in the D^{-1} inner product: numerator is the descent term g^T D^{-1} m, and m=0 is exactly self-aligned
                    d_sqrt = denominator.sqrt()
                    a, b = g / d_sqrt, m_probe / d_sqrt
                elif (
                    align == "white"
                ):  # comparing \(D^{-1}g\) with \(D^{-1}m\), whose dot product can have the opposite sign from the actual descent term \(g^\top D^{-1}m\)
                    a, b = g / denominator, u_probe
                else:  # align == "moment":
                    a, b = g, m_probe

                a_norm, b_norm, beta1_eff = get_norms_and_eff_beta(
                    a,
                    b,
                    pwr,
                )
                beta1_eff = _apply_gate(beta1, beta1_eff, gate_mode)
                # a vanishes iff g vanishes, so the zero-gradient fallback can guard on a_norm
                beta1_eff = torch.where(a_norm > 0.0, beta1_eff, beta1)

                # Gradient weight pinned at (1-beta1); the gate touches memory only (out-of-place: buffer untouched)
                m_eff = m.mul(beta1_eff).add_(g, alpha=(1.0 - beta1))
                r_eff = beta1_eff * r + (1.0 - beta1)

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
                    m_eff_norm = torch.linalg.vector_norm(m_eff).clamp_min(eps)
                    m_eff_unbias = (m_eff / r_probe) * (m_probe_norm / m_eff_norm)
                    u_eff = m_eff_unbias.div_(denominator)

                else:
                    u_eff = (m_eff / r_eff).div_(denominator)
                    if scale == "step":
                        # ||u_probe|| must come from the probe *step*, not the align pair (b is not u_probe under "moment"/"metric")
                        u_probe_norm = b_norm if align in ("update", "white") else torch.linalg.vector_norm(u_probe)
                        u_eff_norm = torch.linalg.vector_norm(u_eff).clamp_min(eps)
                        u_eff.mul_(u_probe_norm / u_eff_norm)

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay

                p.sub_(u_eff, alpha=lr)

        return loss
