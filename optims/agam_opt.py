import copy
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer


def get_alignment_stats(
    g: torch.Tensor,
    probe: torch.Tensor,
    pwr: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the two norms, clamped cosine, and transformed alignment gate."""
    g_norm = torch.linalg.vector_norm(g)
    probe_norm = torch.linalg.vector_norm(probe)
    dot = (g * probe).sum()

    denominator = g_norm.clamp_min(eps) * probe_norm.clamp_min(eps)
    cosine_sim = (dot / denominator).clamp(-1.0, 1.0)

    return g_norm, probe_norm, cosine_sim, ((1.0 + cosine_sim) * 0.5) ** pwr


def get_norms_and_eff_beta(
    g: torch.Tensor,
    probe: torch.Tensor,
    pwr: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility helper returning the historical norm/norm/gate triple."""
    g_norm, probe_norm, _cosine_sim, gate = get_alignment_stats(g, probe, pwr, eps)
    return g_norm, probe_norm, gate


def _apply_gate(base_beta: float, gate: torch.Tensor, gate_mode: str) -> torch.Tensor:
    """Map an alignment gate in [0, 1] to the memory coefficient."""
    if gate_mode == "attenuate":
        return gate.mul(base_beta)
    if gate_mode == "replace":
        return gate
    raise ValueError(f"Invalid gate_mode value: {gate_mode}")


def _copy_state_dict_for_migration(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Copy checkpoint containers without duplicating potentially large tensors."""
    migrated = state_dict.copy()
    migrated["param_groups"] = copy.deepcopy(state_dict["param_groups"])
    migrated["state"] = {
        parameter_id: parameter_state.copy() if isinstance(parameter_state, dict) else parameter_state
        for parameter_id, parameter_state in state_dict.get("state", {}).items()
    }
    return migrated


class AGAM_SGD(Optimizer):
    r"""Alignment-GAted heavy-ball SGD.

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

    ``gradient_weight_mode="complement"`` provides the QHM-like alternative

    :math:`m_t^{eff}=a_t m_{t-1}+(1-a_t)g_t`,

    where :math:`a_t=\beta q_t`.  Its transient buffer is the normalized EMA
    :math:`m_t^{probe}=\beta m_{t-1}+(1-\beta)g_t`; multiplying every stored
    buffer by a positive constant would not change the cosine gate, but the
    normalized form makes the complementary weights meaningful.  The three
    ``unbias`` modes then apply, respectively:

    - ``"none"``: the raw adaptive EMA;
    - ``"buffer"``: :math:`q_t\hat m_t+(1-q_t)g_t`, where
      :math:`\hat m_t=m_t^{probe}/(1-\beta^t)` is the unbiased base EMA;
    - ``"estimator"``: divide the adaptive EMA by its exact transient
      coefficient mass :math:`1-a_t\beta^{t-1}`.

    The two corrections are distinct, defensible QHM-style estimators.  They
    require transient (``in_place=False``), unscaled, non-Nesterov updates;
    recursive memory would instead require tracking the entire realized
    coefficient product.  Complementary weighting is restricted to attenuation
    because replacement makes the self-aligned first update identically zero.

    A zero buffer makes the probe self-aligned (:math:`q_t=1`). A zero gradient
    carries no alignment evidence, so :math:`c_t` falls back to the fixed ``beta``
    rather than treating an undefined direction as an artificial 0.5.

    Alignment is measured once over each complete parameter tensor, producing one
    scalar gate per tensor and optimizer step.

    ``gate_observer``, when supplied, receives ``(parameter, cosine, q_t, c_t,
    gradient_norm, probe_norm)`` immediately before the update. These are the
    actual on-device scalars, including the clamped cosine used by MAL and the
    zero-gradient fallback in ``c_t``; the observer must treat them as read-only.
    Parameters with ``grad=None`` produce no observation. This runtime callback
    is excluded from state dicts.

    ``alignment_observer`` additionally receives ``(parameter, gradient,
    old_memory, probe_moment, gated_moment)`` before the stored buffer changes.
    It must immediately reduce these read-only vectors, not retain mutable
    references. Probe/gated moments equal the update directions under the
    default non-Nesterov, unscaled, uncorrected formulation used by telemetry.
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
        gradient_weight_mode: str = "fixed",
        unbias: str = "none",
        *,
        gate_observer: Callable[[torch.nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None] | None = None,
        alignment_observer: Callable[[torch.nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None] | None = None,
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
        if gradient_weight_mode not in ("fixed", "complement"):
            raise ValueError(f"Invalid gradient_weight_mode value: {gradient_weight_mode}")
        if unbias not in ("none", "buffer", "estimator"):
            raise ValueError(f"Invalid unbias value: {unbias}")
        if gradient_weight_mode == "complement" and gate_mode != "attenuate":
            raise ValueError('gradient_weight_mode="complement" requires gate_mode="attenuate"')
        if gradient_weight_mode == "complement" and nesterov:
            raise ValueError('gradient_weight_mode="complement" does not define a Nesterov variant')
        if gradient_weight_mode == "fixed" and unbias != "none":
            raise ValueError('unbias requires gradient_weight_mode="complement"')
        if unbias != "none" and in_place:
            raise ValueError('unbias requires in_place=False because the correction assumes transient base memory')
        if unbias != "none" and scale:
            raise ValueError('unbias requires scale=False; tensorwise norm matching would cancel its scalar correction')

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
            "gradient_weight_mode": gradient_weight_mode,
            "unbias": unbias,
        }  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute
        self.gate_observer = gate_observer
        # Read-only runtime diagnostics; never serialized in optimizer state.
        # Receives (parameter, consumed gradient, old memory, probe, gated moment)
        # before memory is overwritten. The latter two are the actual update
        # directions for the default, unscaled, non-Nesterov formulation.
        self.alignment_observer = alignment_observer

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load current checkpoints and migrate the former group-list layout."""
        migrated = _copy_state_dict_for_migration(state_dict)
        state = migrated.setdefault("state", {})
        for group in migrated["param_groups"]:
            removed_safeguard = group.pop("descent_safeguard", False)
            if removed_safeguard:
                raise ValueError("Cannot load a checkpoint with the removed descent safeguard enabled.")
            if group.get("gate_mode", self.defaults["gate_mode"]) not in ("replace", "attenuate"):
                raise ValueError(f"Unsupported MAL-SGDM gate_mode in checkpoint: {group.get('gate_mode')}")

            legacy_momentum = group.pop("momentum", None)
            if legacy_momentum is not None:
                if len(legacy_momentum) != len(group["params"]):
                    raise ValueError("Legacy MAL-SGDM checkpoint has inconsistent momentum state.")
                for parameter_id, momentum_buffer in zip(group["params"], legacy_momentum, strict=True):
                    state.setdefault(parameter_id, {})["momentum_buffer"] = momentum_buffer

            # Missing fields identify a pre-QHM checkpoint, whose recurrence
            # was necessarily the historical fixed-gradient formulation.
            group.setdefault("gradient_weight_mode", "fixed")
            group.setdefault("unbias", "none")
            for key, default in self.defaults.items():
                group.setdefault(key, default)

            gradient_weight_mode = group["gradient_weight_mode"]
            unbias = group["unbias"]
            if gradient_weight_mode not in ("fixed", "complement"):
                raise ValueError(f"Checkpoint has unsupported gradient_weight_mode: {gradient_weight_mode}")
            if unbias not in ("none", "buffer", "estimator"):
                raise ValueError(f"Checkpoint has unsupported unbias mode: {unbias}")
            if gradient_weight_mode == "complement" and group["gate_mode"] != "attenuate":
                raise ValueError('gradient_weight_mode="complement" requires gate_mode="attenuate"')
            if gradient_weight_mode == "complement" and group["nesterov"]:
                raise ValueError('gradient_weight_mode="complement" does not define a Nesterov variant')
            if gradient_weight_mode == "fixed" and unbias != "none":
                raise ValueError('unbias requires gradient_weight_mode="complement"')
            if unbias != "none" and group["in_place"]:
                raise ValueError('unbias requires in_place=False because the correction assumes transient base memory')
            if unbias != "none" and group["scale"]:
                raise ValueError('unbias requires scale=False; tensorwise norm matching would cancel its scalar correction')

            for parameter_id in group["params"]:
                parameter_state = state.get(parameter_id)
                if not isinstance(parameter_state, dict) or "momentum_buffer" not in parameter_state:
                    continue
                # ``t`` existed briefly during development; published state
                # uses the same ``step`` spelling as the other optimizers.
                parameter_state["step"] = int(parameter_state.pop("t", parameter_state.get("step", 0)))
        super().load_state_dict(migrated)

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
            gradient_weight_mode = group["gradient_weight_mode"]
            unbias = group["unbias"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]  # used such that loading model form checkpoint pushes all its weights + states to correct device
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                state.setdefault("step", 0)

                state["step"] += 1
                step = state["step"]

                m = state["momentum_buffer"]
                if wd > 0.0:
                    # Coupled weight decay
                    g = g.add(p, alpha=wd)

                if gradient_weight_mode == "fixed":
                    m_probe = g.add(m, alpha=beta)
                else:
                    m_probe = g.lerp(m, weight=beta)

                u = g.add(m_probe, alpha=beta) if nesterov else m_probe

                g_norm, u_norm, cosine_sim, gate = get_alignment_stats(g, u, pwr)
                beta_eff = _apply_gate(beta, gate, gate_mode)
                beta_eff = torch.where(g_norm > 0.0, beta_eff, beta)
                if self.gate_observer is not None:
                    self.gate_observer(p, cosine_sim, gate, beta_eff, g_norm, u_norm)

                if gradient_weight_mode == "fixed":
                    m_eff = g.addcmul(m, beta_eff)  # beta_eff * m_{t-1} + g
                else:
                    m_eff = g.lerp(m, weight=beta_eff)  # beta_eff * m_{t-1} + (1 - beta_eff) g

                if self.alignment_observer is not None:
                    self.alignment_observer(p, g, m, m_probe, m_eff)

                if in_place:
                    m.copy_(m_eff)
                else:
                    m.copy_(m_probe)  # plain heavy ball

                u_eff = g.addcmul(m, beta_eff) if nesterov else m_eff

                if scale:
                    u_eff_norm = torch.linalg.vector_norm(u_eff) + 1e-8
                    u_eff.mul_(u_norm.clamp_min(1e-12) / u_eff_norm)

                if unbias == "buffer":
                    beta_power = beta**step
                    num_term = g * (gate - 1.0) * beta_power
                    u_eff.add_(num_term).div_(1.0 - beta_power)
                elif unbias == "estimator":
                    # The fallback for a zero gradient sets beta_eff=beta even
                    # though its undefined raw cosine maps to gate=0.5.  Using
                    # beta_eff here keeps the coefficient mass exact in that
                    # case as well as for ordinary non-zero gradients.
                    coefficient_mass = 1.0 - beta_eff * beta ** (step - 1)
                    u_eff.div_(coefficient_mass)

                p.sub_(u_eff, alpha=lr)

        return loss


class AGAM_AdamW(Optimizer):
    r"""Alignment-GAted AdamW.

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

    ``first_moment_correction="adaptive"`` applies this exact realized
    coefficient mass.  For the default transient complementary update it is
    :math:`r_t^{eff}=1-q_t\beta_1^t`.  The explicit ablation
    ``first_moment_correction="standard"`` instead divides by AdamW's fixed
    :math:`1-\beta_1^t`.  It is restricted to ``scale="none"`` because scalar
    correction is canceled by either norm-matching mode.

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
        align: str = "update",
        in_place: bool = False,
        scale: bool | str = False,
        gate_mode: str = "attenuate",
        gradient_weight_mode="complement",
        first_moment_correction: str = "adaptive",
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
        if first_moment_correction not in ("adaptive", "standard"):
            raise ValueError(f"Invalid first_moment_correction value: {first_moment_correction}")
        if first_moment_correction == "standard" and scale != "none":
            raise ValueError('first_moment_correction="standard" requires scale="none"')

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
            "first_moment_correction": first_moment_correction,
        }
        super().__init__(optim_groups, defaults)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load current checkpoints and migrate the former group-list layout."""
        migrated = _copy_state_dict_for_migration(state_dict)
        state = migrated.setdefault("state", {})
        legacy_state_keys = {
            "m": "exp_avg",
            "v": "exp_avg_sq",
            "r": "first_moment_weight",
            "step": "step",
        }
        for group in migrated["param_groups"]:
            removed_safeguard = group.pop("descent_safeguard", False)
            if removed_safeguard:
                raise ValueError("Cannot load a checkpoint with the removed descent safeguard enabled.")

            legacy_values = {key: group.pop(key, None) for key in legacy_state_keys}
            present_legacy_keys = {key for key, values in legacy_values.items() if values is not None}
            if present_legacy_keys and present_legacy_keys != set(legacy_state_keys):
                raise ValueError("Legacy MAL-AdamW checkpoint has incomplete optimizer state.")
            if present_legacy_keys:
                if any(len(values) != len(group["params"]) for values in legacy_values.values()):
                    raise ValueError("Legacy MAL-AdamW checkpoint has inconsistent optimizer state.")
                for index, parameter_id in enumerate(group["params"]):
                    parameter_state = state.setdefault(parameter_id, {})
                    for legacy_key, state_key in legacy_state_keys.items():
                        parameter_state[state_key] = legacy_values[legacy_key][index]

            group.setdefault("gradient_weight_mode", "fixed")
            gradient_weight_mode = group["gradient_weight_mode"]
            if gradient_weight_mode not in ("fixed", "complement"):
                raise ValueError(f"Checkpoint has unsupported gradient_weight_mode: {gradient_weight_mode}")
            gate_mode = group.get("gate_mode", self.defaults["gate_mode"])
            if gate_mode not in ("replace", "attenuate"):
                raise ValueError(f"Unsupported MAL-AdamW gate_mode in checkpoint: {gate_mode}")
            if gradient_weight_mode == "complement" and gate_mode != "attenuate":
                raise ValueError('gradient_weight_mode="complement" requires gate_mode="attenuate"')

            scale = group.get("scale", self.defaults["scale"])
            if isinstance(scale, bool):
                group["scale"] = "step" if scale else "none"
            elif scale not in ("step", "moment", "none"):
                raise ValueError(f"Unsupported MAL-AdamW scale in checkpoint: {scale}")
            first_moment_correction = group.get(
                "first_moment_correction", self.defaults["first_moment_correction"]
            )
            if first_moment_correction not in ("adaptive", "standard"):
                raise ValueError(
                    "Checkpoint has unsupported first_moment_correction: "
                    f"{first_moment_correction}"
                )
            if first_moment_correction == "standard" and group["scale"] != "none":
                raise ValueError('first_moment_correction="standard" requires scale="none"')
            if group.get("align", self.defaults["align"]) not in ("update", "metric", "white", "moment"):
                raise ValueError(f"Unsupported MAL-AdamW align in checkpoint: {group.get('align')}")

            for key, default in self.defaults.items():
                group.setdefault(key, default)
        super().load_state_dict(migrated)

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
            first_moment_correction = group["first_moment_correction"]
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
                    grad, base_step = g, u
                elif align == "metric":  # cosine in the D^{-1} inner product: numerator is the descent term g^T D^{-1} m, and m=0 is exactly self-aligned
                    d_sqrt = denominator.sqrt()
                    grad, base_step = g / d_sqrt, m_probe / d_sqrt
                elif (
                    align == "white"
                ):  # comparing \(D^{-1}g\) with \(D^{-1}m\), whose dot product can have the opposite sign from the actual descent term \(g^\top D^{-1}m\)
                    grad, base_step = g / denominator, u
                else:  # align == "moment":
                    grad, base_step = g, m_probe

                a_norm, b_norm, beta1_eff = get_norms_and_eff_beta(
                    grad,
                    base_step,
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
                    first_moment_divisor = (
                        r_eff
                        if first_moment_correction == "adaptive"
                        else 1.0 - beta1**step
                    )
                    u_eff = (m_eff / first_moment_divisor).div_(denominator)
                    if scale == "step":
                        # ||u|| must come from the probe *step*, not the align pair (b is not u under "moment"/"metric")
                        u_probe_norm = b_norm if align in ("update", "white") else torch.linalg.vector_norm(u)
                        u_eff_norm = torch.linalg.vector_norm(u_eff) + eps
                        u_eff.mul_(u_probe_norm / u_eff_norm)

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay

                p.sub_(u_eff, alpha=lr)

        return loss


# class AdaAGAM(Optimizer):
#     """AGAM heavy-ball numerator with adaptive squared-gradient scaling.
#
#     Each tensor's cosine gates its historical contribution, giving
#     ``m_eff = beta_eff * m + g``. With ``m_probe = beta1 * m + g``, alignment is
#     ``cos(g, m_probe)`` for ``align="moment"`` (the default),
#     ``cos(g, m_probe / D)`` for ``align="update"``, or
#     ``cos(g / sqrt(D), m_probe / sqrt(D))`` for ``align="metric"``, using the
#     current adaptive denominator D. Both adaptive geometries have numerator
#     ``g.T @ inv(D) @ m_probe``, the first-order descent term of the probe;
#     ``metric`` additionally uses the norm induced by ``inv(D)``. The first
#     moment is never normalized. ``unbias`` controls only the second moment;
#     its default is False, matching AdaTAM's denominator.
#
#     Transient state stores the fixed-beta probe. Recursive state stores the
#     effective moment; with ``scale="moment"`` it deliberately stores the
#     rescaled moment. ``scale="step"`` rescales only the applied update, so it
#     leaves that recursive moment unscaled. Norm matching is approximate due
#     to the additive numerical epsilon.
#     """
#
#     def __init__(
#         self,
#         params: Iterable[torch.nn.Parameter],
#         lr: float = 1e-3,
#         betas: tuple[float, float] = (0.9, 0.95),
#         eps: float = 1e-8,
#         weight_decay: float = 0.0,
#         pwr: float = 1.0,
#         in_place: bool = False,
#         scale: bool | str = False,
#         gate_mode: str = "attenuate",
#         unbias: bool = False,
#         align: str = "moment",
#     ) -> None:
#         if lr < 0.0:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError(f"Invalid beta1 value: {betas[0]}")
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError(f"Invalid beta2 value: {betas[1]}")
#         if eps <= 0.0:
#             raise ValueError(f"Invalid eps value: {eps}")
#         if weight_decay < 0.0:
#             raise ValueError(f"Invalid weight_decay value: {weight_decay}")
#         if pwr not in (0.5, 1.0):
#             raise ValueError(f"Invalid p value: {pwr}")
#         if isinstance(scale, bool):
#             scale = "step" if scale else "none"
#         if scale not in ("step", "moment", "none"):
#             raise ValueError(f"Invalid scale value: {scale}")
#         if gate_mode not in ("replace", "attenuate"):
#             raise ValueError(f"Invalid gate_mode value: {gate_mode}")
#         if not isinstance(unbias, bool):
#             raise TypeError(f"Invalid unbias value: {unbias}")
#         if align not in ("moment", "update", "metric"):
#             raise ValueError(f"Invalid AdaMAL align value: {align}")
#
#         decay_params: list[torch.nn.Parameter] = []
#         no_decay_params: list[torch.nn.Parameter] = []
#
#         for p in params:
#             if not p.requires_grad:
#                 continue
#             # Exclude biases and 1D normalization parameters from weight decay
#             if weight_decay == 0 or p.ndim <= 1:
#                 no_decay_params.append(p)
#             else:
#                 decay_params.append(p)
#
#         if not decay_params and not no_decay_params:
#             raise ValueError("AdamW received no trainable parameters.")
#
#         optim_groups = []
#
#         for group_params, group_wd in ((no_decay_params, 0.0), (decay_params, weight_decay)):
#             if group_params:
#                 optim_groups.append(
#                     {
#                         "params": group_params,
#                         "weight_decay": group_wd,
#                     }
#                 )
#
#         defaults = {
#             "lr": lr,
#             "beta1": betas[0],
#             "beta2": betas[1],
#             "pwr": pwr,
#             "eps": eps,
#             "in_place": in_place,
#             "scale": scale,
#             "gate_mode": gate_mode,
#             "unbias": unbias,
#             "align": align,
#         }
#         super().__init__(optim_groups, defaults)
#
#         # Model-wide, parameter-count-weighted diagnostics from the latest
#         # optimizer step. They remain on-device until the training loop logs
#         # them, avoiding per-parameter host synchronization.
#         self.last_gate_mean: torch.Tensor | None = None
#         self.last_gate_min: torch.Tensor | None = None
#         self.last_gate_max: torch.Tensor | None = None
#         self.last_beta_eff_mean: torch.Tensor | None = None
#
#     @property
#     def unbias(self) -> bool:
#         """Whether the adaptive second moment is bias-corrected."""
#         values = {group["unbias"] for group in self.param_groups}
#         if len(values) != 1:
#             raise RuntimeError("AdaMAL parameter groups disagree on unbias.")
#         return values.pop()
#
#     @unbias.setter
#     def unbias(self, value: bool) -> None:
#         if not isinstance(value, bool):
#             raise TypeError(f"Invalid unbias value: {value}")
#         for group in self.param_groups:
#             group["unbias"] = value
#
#     def load_state_dict(self, state_dict: dict[str, Any]) -> None:
#         """Load AdaMAL state while preserving historical configuration meaning."""
#         migrated = _copy_state_dict_for_migration(state_dict)
#         for group in migrated["param_groups"]:
#             # AdaMAL checkpoints created before alignment was configurable used
#             # raw-moment alignment exclusively.
#             group.setdefault("align", "moment")
#             group.setdefault("unbias", self.defaults["unbias"])
#             for key, default in self.defaults.items():
#                 group.setdefault(key, default)
#
#             if group["align"] not in ("moment", "update", "metric"):
#                 raise ValueError(f"Checkpoint has unsupported AdaMAL align: {group['align']}")
#             if group["pwr"] not in (0.5, 1.0):
#                 raise ValueError(f"Checkpoint has unsupported AdaMAL pwr: {group['pwr']}")
#             if group["gate_mode"] not in ("replace", "attenuate"):
#                 raise ValueError(f"Checkpoint has unsupported AdaMAL gate_mode: {group['gate_mode']}")
#             scale = group["scale"]
#             if isinstance(scale, bool):
#                 group["scale"] = "step" if scale else "none"
#             elif scale not in ("none", "step", "moment"):
#                 raise ValueError(f"Checkpoint has unsupported AdaMAL scale: {scale}")
#             if not isinstance(group["unbias"], bool):
#                 raise TypeError(f"Checkpoint has invalid AdaMAL unbias: {group['unbias']}")
#         super().load_state_dict(migrated)
#
#     @torch.no_grad()
#     def step(self, closure: Callable[[], float | torch.Tensor] | None = None) -> Any:
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#
#         gate_total: torch.Tensor | None = None
#         beta_eff_total: torch.Tensor | None = None
#         gate_min: torch.Tensor | None = None
#         gate_max: torch.Tensor | None = None
#         diagnostic_parameter_count = 0
#
#         for group in self.param_groups:
#             lr = group["lr"]
#             wd = group["weight_decay"]
#             beta1 = group["beta1"]
#             beta2 = group["beta2"]
#             pwr = group["pwr"]
#             in_place = group["in_place"]
#             scale = group["scale"]
#             gate_mode = group["gate_mode"]
#             eps = group["eps"]
#             unbias = group["unbias"]
#             align = group["align"]
#
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#
#                 g = p.grad
#                 if g.is_sparse:
#                     raise RuntimeError("AdaMAL does not support sparse gradients")
#                 if torch.is_complex(p) or torch.is_complex(g):
#                     raise RuntimeError("AdaMAL does not support complex parameters or gradients")
#                 state = self.state[p]
#                 if not state:
#                     state["step"] = 0
#                     state["momentum_buffer"] = torch.zeros_like(p)
#                     state["exp_avg_sq"] = torch.zeros_like(p)
#
#                 state["step"] += 1
#                 step = state["step"]
#                 m = state["momentum_buffer"]
#                 v = state["exp_avg_sq"]
#
#                 m_probe = g.add(m, alpha=beta1)
#                 v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
#                 if unbias:
#                     v_unbias = v / (1.0 - beta2**step)
#                 else:
#                     v_unbias = v
#                 denominator = v_unbias.sqrt().add_(eps)
#
#                 if align == "update":
#                     alignment_gradient = g
#                     alignment_probe = m_probe.div(denominator)
#                 elif align == "metric":
#                     metric_sqrt = denominator.sqrt()
#                     alignment_gradient = g.div(metric_sqrt)
#                     alignment_probe = m_probe.div(metric_sqrt)
#                 else:
#                     alignment_gradient = g
#                     alignment_probe = m_probe
#
#                 g_norm, alignment_probe_norm, gate = get_norms_and_eff_beta(
#                     alignment_gradient,
#                     alignment_probe,
#                     pwr,
#                 )
#                 beta1_eff = _apply_gate(beta1, gate, gate_mode)
#                 beta1_eff = torch.where(g_norm > 0.0, beta1_eff, beta1)
#                 diagnostic_weight = p.numel()
#                 weighted_gate = gate * diagnostic_weight
#                 weighted_beta_eff = beta1_eff * diagnostic_weight
#                 gate_total = weighted_gate if gate_total is None else gate_total + weighted_gate
#                 beta_eff_total = weighted_beta_eff if beta_eff_total is None else beta_eff_total + weighted_beta_eff
#                 gate_min = gate if gate_min is None else torch.minimum(gate_min, gate)
#                 gate_max = gate if gate_max is None else torch.maximum(gate_max, gate)
#                 diagnostic_parameter_count += diagnostic_weight
#                 m_eff = g.addcmul(m, beta1_eff)
#
#                 if scale == "moment":
#                     m_probe_norm = alignment_probe_norm if align == "moment" else torch.linalg.vector_norm(m_probe)
#                     m_eff_norm = torch.linalg.vector_norm(m_eff) + eps
#                     m_eff.mul_(m_probe_norm.clamp_min(1e-12) / m_eff_norm)  # commit scaled version into buffer if in-place is True
#                     u_eff = m_eff.div(denominator)
#
#                 else:
#                     u_eff = m_eff.div(denominator)
#                     if scale == "step":
#                         u_probe_norm = alignment_probe_norm if align == "update" else torch.linalg.vector_norm(m_probe.div(denominator))
#                         u_eff_norm = torch.linalg.vector_norm(u_eff) + eps
#                         u_eff.mul_(u_probe_norm / u_eff_norm)
#
#                 if in_place:
#                     m.copy_(m_eff)
#                 else:
#                     m.copy_(m_probe)
#
#                 if wd > 0.0:
#                     p.mul_(1.0 - lr * wd)  # decoupled decay
#
#                 p.sub_(u_eff, alpha=lr)
#
#         if gate_total is not None:
#             self.last_gate_mean = gate_total / diagnostic_parameter_count
#             self.last_gate_min = gate_min
#             self.last_gate_max = gate_max
#             assert beta_eff_total is not None
#             self.last_beta_eff_mean = beta_eff_total / diagnostic_parameter_count
#
#         return loss
