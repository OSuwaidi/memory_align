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

                mask = (m * g) > 0.0
                scale = mask.numel() / (mask.sum() + 1.0)
                scaled_mask = mask.to(m.dtype).mul_(scale)

                p.addcmul_(m, scaled_mask, value=-lr)
