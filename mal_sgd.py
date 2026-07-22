import warnings
from typing import Iterable

import torch
from torch.optim import Optimizer


class MAL_SGD(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 0.1,
            beta: float = 0.9,
            weight_decay: float = 0.0,
            couple: bool = True,
            mem_align: bool = True,
            tau: float = 0.0,
            ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= tau < 1.0:
            raise ValueError(f"Invalid tau value: {tau}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.couple = couple
        self.mem_align = mem_align
        self.tau = tau

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

        defaults = dict(lr=lr, beta=beta)  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)  # exposes "self.param_groups" attribute

    @staticmethod
    def _params_to_vec(
            params: Iterable[torch.nn.Parameter | torch.Tensor],
            ) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in params])

    @staticmethod
    def _param_grads_to_vec(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        return torch.cat(
                [
                    p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1)
                    for p in params
                    ]
                )

    @staticmethod
    def _assign_vec_to_params(
            vec: torch.Tensor, params: Iterable[torch.nn.Parameter],
            ) -> None:
        pointer = 0
        for param in params:
            end = pointer + param.numel()
            param.copy_(vec[pointer:end].view_as(param))
            pointer = end

    @staticmethod
    def layer_bounce(G1: torch.Tensor, G2: torch.Tensor, tau: float = 0.0) -> torch.Tensor:
        # If gradients are *misaligned* ==>, their dot product is negative
        return (G1 @ G2) < (-tau * G1.norm() * G2.norm())

    @staticmethod
    def per_bounce(G1: torch.Tensor, G2: torch.Tensor, ) -> torch.Tensor:
        return (G1.mul(G2)) < 0.0

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["beta"]

            for p, m in zip(group["params"], group["momentum"]):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    if self.couple:
                        g.add_(p, alpha=wd)
                    else:
                        p.mul_(1.0 - lr * wd)

                # Absorb current gradient into momentum:
                m.mul_(beta).add_(g)

                if self.mem_align:
                    bounce_cond = self.layer_bounce(m.view(-1), g.view(-1), self.tau).to(g.dtype)
                    m.lerp_(g, weight=bounce_cond)

                p.sub_(m, alpha=lr)
