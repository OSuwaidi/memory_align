"""Lion and its Alignment-GAted Momentum wrapper.

The base implementation follows the official Google Research PyTorch update:

``update = sign(beta1 * exp_avg + (1 - beta1) * gradient)``
``exp_avg = beta2 * exp_avg + (1 - beta2) * gradient``

AGAM-Lion changes only the coefficient used by the first, immediately applied
mixture.  The persistent ``exp_avg`` advances exactly as it does in Lion.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    r"""EvoLved Sign Momentum (Lion).

    ``beta1`` mixes the historical EMA with the current gradient for the
    immediately applied sign update. ``beta2`` advances the persistent EMA.
    Weight decay is decoupled, matching the reference implementation.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    def _effective_beta1(
        self,
        gradient: torch.Tensor,
        exp_avg: torch.Tensor,
        group: dict[str, Any],
    ) -> float | torch.Tensor:
        del gradient, exp_avg
        return group["betas"][0]

    @torch.no_grad()
    def step(self, closure: Callable[[], float | torch.Tensor] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta2 = group["betas"][1]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError(f"{type(self).__name__} does not support sparse gradients")
                if torch.is_complex(parameter) or torch.is_complex(gradient):
                    raise RuntimeError(f"{type(self).__name__} does not support complex parameters or gradients")

                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]

                beta1_eff = self._effective_beta1(gradient, exp_avg, group)
                immediate_mixture = exp_avg * beta1_eff + gradient * (1.0 - beta1_eff)

                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(immediate_mixture.sign(), alpha=-lr)

                # AGAM-Lion deliberately leaves this base-Lion state recurrence
                # untouched; only the immediately applied beta1 is gated.
                exp_avg.mul_(beta2).add_(gradient, alpha=1.0 - beta2)

        return loss


class AGAM_Lion(Lion):
    r"""Alignment-GAted Lion.

    For each parameter tensor, first form Lion's base applied direction

    .. math::
        \hat u_t=\operatorname{sign}(\beta_1m_{t-1}+(1-\beta_1)g_t).

    The tensor-wise gate and effective immediate-memory coefficient are

    .. math::
        \gamma_t=\left(\frac{1+\cos(g_t,\hat u_t)}{2}\right)^p,
        \qquad \beta_{1,t}^{\mathrm{eff}}=\gamma_t\beta_1.

    AGAM-Lion then applies

    .. math::
        \operatorname{sign}(\beta_{1,t}^{\mathrm{eff}}m_{t-1}
        +(1-\beta_{1,t}^{\mathrm{eff}})g_t),

    while the stored EMA remains exactly Lion's
    :math:`m_t=\beta_2m_{t-1}+(1-\beta_2)g_t`.  Thus this is a transient,
    complementary gate on Lion's immediate ``beta1`` mixture, not a change to
    its long-term state. Undefined alignment (a zero gradient or zero probe)
    falls back to the base ``beta1``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        pwr: float = 1.0,
        alignment_eps: float = 1e-8,
    ) -> None:
        if pwr <= 0.0:
            raise ValueError(f"Invalid pwr value: {pwr}")
        if alignment_eps <= 0.0:
            raise ValueError(f"Invalid alignment_eps value: {alignment_eps}")
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.defaults["pwr"] = pwr
        self.defaults["alignment_eps"] = alignment_eps
        for group in self.param_groups:
            group.setdefault("pwr", pwr)
            group.setdefault("alignment_eps", alignment_eps)

    def _effective_beta1(
        self,
        gradient: torch.Tensor,
        exp_avg: torch.Tensor,
        group: dict[str, Any],
    ) -> torch.Tensor:
        beta1 = group["betas"][0]
        probe_direction = (exp_avg * beta1 + gradient * (1.0 - beta1)).sign()
        gradient_norm = torch.linalg.vector_norm(gradient)
        probe_norm = torch.linalg.vector_norm(probe_direction)
        denominator = gradient_norm.clamp_min(group["alignment_eps"]) * probe_norm.clamp_min(group["alignment_eps"])
        cosine = ((gradient * probe_direction).sum() / denominator).clamp(-1.0, 1.0)
        gate = ((1.0 + cosine) * 0.5).pow(group["pwr"])
        beta1_eff = gate * beta1
        has_alignment_evidence = (gradient_norm > 0.0) & (probe_norm > 0.0)
        return torch.where(has_alignment_evidence, beta1_eff, beta1_eff.new_tensor(beta1))
