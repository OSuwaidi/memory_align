import warnings
from typing import Iterable

import torch
from torch.optim import Optimizer


class CAUTIOUS_SGD(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 0.1,
            beta: float = 0.9,
            weight_decay: float = 0.0,
            nesterov: bool=False,
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
        decay_momentum = []

        no_decay_params: list[torch.nn.Parameter] = []
        no_decay_momentum = []

        for p in params:
            if not p.requires_grad:
                continue
            device = p.device
            # Exclude biases and 1D normalization parameters from weight decay
            if weight_decay == 0 or p.ndim <= 1:
                no_decay_params.append(p)
                no_decay_momentum.append(torch.zeros_like(p))
            else:
                decay_params.append(p)
                decay_momentum.append(torch.zeros_like(p))

        if not decay_params and not no_decay_params:
            raise ValueError("SGD received no trainable parameters.")

        if "cuda" not in device.type:
            warnings.warn(
                    f"Model parameters' device is not CUDA, rather is {device.type}!",
                    stacklevel=2,
                    )

        optim_groups = []

        if no_decay_params:
            optim_groups.append(
                    {
                        "params": no_decay_params,
                        "momentum": no_decay_momentum,
                        "weight_decay": 0.0,
                        }
                    )
        if decay_params:
            optim_groups.append(
                    {
                        "params": decay_params,
                        "momentum": decay_momentum,
                        "weight_decay": weight_decay,
                        },
                    )

        defaults = dict(lr=lr, beta=beta, nesterov=nesterov)  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["beta"]
            nesterov = group["nesterov"]

            for p, m in zip(group["params"], group["momentum"]):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    g.add_(p, alpha=wd)

                # Absorb current gradient into momentum:
                m.mul_(beta).add_(g)

                # The cautious mask applies to the APPLIED update (the NAG look-ahead
                # u = g + beta*m under Nesterov); the momentum buffer is never masked
                u = torch.add(g, m, alpha=beta) if nesterov else m

                mask = (u * g) > 0.0
                scale = mask.numel() / (mask.sum() + 1.0)
                scaled_mask = mask.to(u.dtype).mul_(scale)

                p.addcmul_(u, scaled_mask, value=-lr)


class CAUTIOUS_ADAMW(Optimizer):
    """C-AdamW (Cautious Optimizers, arXiv:2411.16085) for transformer training.

    Faithful to the official implementation: the per-coordinate mask (m * g > 0) is
    applied to the UPDATE only (the momentum/variance state is never masked), and the
    surviving update is rescaled by ~1/mean(mask). Since the preconditioner is
    positive, sign(m/denom) == sign(m), so masking m against g is exactly the paper's
    update-vs-gradient criterion. Decoupled weight decay is applied unmasked (AdamW).
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

        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.t = 0  # global step count; assumes every param receives a grad each step

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for p in params:
            if not p.requires_grad:
                continue
            device = p.device
            # Exclude biases and 1D normalization parameters from weight decay
            if weight_decay == 0 or p.ndim <= 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        if not decay_params and not no_decay_params:
            raise ValueError("AdamW received no trainable parameters.")

        if "cuda" not in device.type:
            warnings.warn(
                    f"Model parameters' device is not CUDA, rather is {device.type}!",
                    stacklevel=2,
                    )

        optim_groups = []

        for group_params, group_wd in ((no_decay_params, 0.0), (decay_params, weight_decay)):
            if group_params:
                optim_groups.append(
                        {
                            "params": group_params,
                            "m": [torch.zeros_like(p) for p in group_params],
                            "v": [torch.zeros_like(p) for p in group_params],
                            "weight_decay": group_wd,
                            }
                        )

        defaults = dict(lr=lr)  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2_sqrt = (1.0 - self.beta2 ** self.t) ** 0.5

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            for p, m, v in zip(group["params"], group["m"], group["v"]):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay; applied regardless of the mask

                m.lerp_(g, 1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                mask = (m * g) > 0.0
                scale = mask.numel() / (mask.sum() + 1.0)
                scaled_mask = mask.to(m.dtype).mul_(scale)

                u = m.div(v.sqrt().div_(bc2_sqrt).add_(self.eps)).mul_(scaled_mask)
                p.add_(u, alpha=-lr / bc1)
