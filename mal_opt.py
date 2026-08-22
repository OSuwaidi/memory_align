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
    probe_norm = torch.linalg.vector_norm(probe)
    dot = torch.dot(g.flatten(), probe.flatten())

    denominator = g_norm.clamp_min(eps) * probe_norm.clamp_min(eps)
    cosine_sim = (dot / denominator).clamp(-1.0, 1.0)

    return g_norm, probe_norm, ((1.0 + cosine_sim) * 0.5) ** pwr  # effective momentum coefficient for this step


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
        scale: bool = True,
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
                    # Coupled weight decay
                    g = g.add(p, alpha=wd)

                probe_m = g.add(m, alpha=beta)
                probe_u = g.add(probe_m, alpha=beta) if nesterov else probe_m

                # Fan-in axes for per-output-unit alignment; None keeps the
                # whole-tensor cosine (also the only sane choice for ndim <= 1,
                # where each "unit" is a lone scalar).
                dims = tuple(range(1, p.ndim)) if (per_unit and p.ndim > 1) else None

                if dims is None:
                    g_norm = torch.linalg.vector_norm(g)
                    probe_norm = torch.linalg.vector_norm(probe_u)
                    dot = torch.dot(g.flatten(), probe_u.flatten())
                else:
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

                u = torch.addcmul(g, m, eff_beta) if nesterov else eff_m

                if scale:
                    u_norm = torch.linalg.vector_norm(u) if dims is None else torch.linalg.vector_norm(u, dim=dims, keepdim=True)
                    u.mul_(probe_norm / u_norm.clamp_min(1e-8))

                p.sub_(u, alpha=lr)

        return loss


class MAL_AdamW(Optimizer):
    r"""Memory-ALigned AdamW.

    ``m`` is AdamW's first-moment state and ``mass`` is the sum of its raw
    gradient weights. The fixed-beta probe is

        probe_m = beta1 * m + (1 - beta1) * g,

    while MAL applies

        eff_m = c * m + (1 - beta1) * g.

    Keeping the fresh-gradient coefficient fixed makes this the Adam-coordinate
    equivalent of MAL-SGDM's ``g + c * momentum``. Both the probe and effective
    first moments are divided by their corresponding weight mass, which gives
    the ordinary AdamW bias correction when ``c == beta1`` and remains valid
    with adaptive in-place memory. The second moment and decoupled weight decay
    remain plain AdamW.
    """

    MAX_BETA1 = 0.999

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        pwr: float = 1.0,
        in_place: bool = False,
        scale: bool = True,
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
                        "m_mass": [
                            torch.zeros(
                                (),
                                dtype=torch.float32,
                                device=p.device,
                            )
                            for p in group_params
                        ],
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
            "in_place": in_place,
            "scale": scale,
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
            in_place = group["in_place"]
            scale = group["scale"]
            eps = group["eps"]
            steps = group["step"]
            masses = group["m_mass"]

            for i, (p, m, v, mass) in enumerate(zip(group["params"], group["m"], group["v"], masses)):
                if p.grad is None:
                    continue

                g = p.grad
                steps[i] += 1

                probe_m = m.lerp(g, weight=(1.0 - beta1))
                probe_mass = mass.mul(beta1).add(1.0 - beta1)
                unbiased_probe_m = probe_m.div(probe_mass.to(probe_m.dtype))

                v.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2))
                unbiased_v = v / (1.0 - beta2 ** steps[i])
                denominator = unbiased_v.sqrt_().add_(eps)

                probe_u = unbiased_probe_m.div_(denominator)

                g_norm, probe_norm, eff_beta1 = get_norms_and_eff_beta(g.div(denominator), probe_u, pwr)
                # TODO: g_norm, probe_norm, eff_beta1 = get_norms_and_eff_beta(g, probe_m, pwr)
                eff_beta1 = torch.where(g_norm > 0.0, eff_beta1, beta1).clamp_max(self.MAX_BETA1)

                eff_m = m.mul(eff_beta1).add_(g, alpha=(1.0 - beta1))  # lerp == False
                eff_mass = mass.mul(eff_beta1.to(mass.dtype)).add(1.0 - beta1)

                # eff_m = m.lerp(g, weight=(1.0 - eff_beta1))  # lerp == True
                # eff_mass = mass.mul(eff_beta1.to(mass.dtype)).add(1.0 - eff_beta1)

                if in_place:
                    m.copy_(eff_m)
                    mass.copy_(eff_mass)
                else:
                    m.copy_(probe_m)
                    mass.copy_(probe_mass)

                unbiased_eff_m = eff_m.div_(eff_mass.to(eff_m.dtype))
                u = unbiased_eff_m.div_(denominator)

                if scale:
                    u_norm = torch.linalg.vector_norm(u)
                    u.mul_(probe_norm / u_norm.clamp_min(1e-8))

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay

                p.sub_(u, alpha=lr)

        return loss
