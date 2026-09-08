import warnings
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer


def _step_as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Optimizer step state must be scalar.")
        value = value.item()
    numeric_value = float(value)
    if not numeric_value.is_integer() or numeric_value < 0.0:
        raise ValueError(f"Optimizer step state must be a non-negative integer, got {value!r}.")
    return int(numeric_value)


class C_SGDM(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta: float = 0.9,
        weight_decay: float = 0.0,
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
            "nesterov": nesterov,
        }  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

    @torch.no_grad()
    def step(self, closure: Callable[[], float | torch.Tensor] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["beta"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                momentum = state["momentum_buffer"]
                if wd > 0.0:
                    g = g.add(p, alpha=wd)

                # Absorb current gradient into momentum:
                momentum.mul_(beta).add_(g)

                # The cautious mask applies to the APPLIED update (the NAG look-ahead
                # u = g + beta*m under Nesterov); the momentum buffer is never masked
                u = torch.add(g, momentum, alpha=beta) if nesterov else momentum

                mask = (u * g) > 0.0
                scale = mask.numel() / (mask.sum() + 1.0)
                scaled_mask = mask.to(u.dtype).mul_(scale)

                p.addcmul_(u, scaled_mask, value=-lr)

        return loss


class C_AdamW(Optimizer):
    """C-AdamW using the original Cautious Optimizers paper recurrence.

    The per-coordinate mask ``m * g > 0`` is applied only to the update; moment
    state is never masked. This preserves the paper-original normalization
    ``numel(mask) / (mask.sum() + 1)``. Later official-code variants use a
    clamped inverse mask mean instead, so this class intentionally claims
    fidelity to the published formula rather than exact parity with those
    later variants. Decoupled weight decay remains unmasked, as in AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
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

        for group_params, group_wd in ((no_decay_params, 0.0), (decay_params, weight_decay)):
            if group_params:
                optim_groups.append(
                    {
                        "params": group_params,
                        "weight_decay": group_wd,
                    }
                )

        defaults = {"lr": lr, "betas": betas, "eps": eps}
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] = _step_as_int(state.get("step", 0)) + 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay; applied regardless of the mask

                exp_avg.lerp_(g, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                mask = (exp_avg * g) > 0.0
                scale = mask.numel() / (mask.sum() + 1.0)
                scaled_mask = mask.to(exp_avg.dtype).mul_(scale)

                bc1 = 1.0 - beta1**step
                bc2_sqrt = (1.0 - beta2**step) ** 0.5
                update = exp_avg.div(exp_avg_sq.sqrt().div_(bc2_sqrt).add_(eps)).mul_(scaled_mask)
                p.add_(update, alpha=-lr / bc1)

        return loss
