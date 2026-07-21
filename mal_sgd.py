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
            adaptive: bool = True,
            ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.couple = couple
        self.adaptive = adaptive

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
                        "beta": [torch.tensor(beta, device=device) for _ in no_decay_params]
                        }
                    )
        if decay_params:
            optim_groups.append(
                    {
                        "params": decay_params,
                        "momentum": decay_momentum,
                        "weight_decay": weight_decay,
                        "beta": [torch.tensor(beta, device=device) for _ in decay_params]
                        },
                    )

        defaults = dict(lr=lr)  # shared across all optim/param groups
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

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            betas = group["beta"]

            for i, (p, m) in enumerate(zip(group["params"], group["momentum"])):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    if self.couple:
                        g.add_(p, alpha=wd)
                    else:
                        p.mul_(1.0 - lr * wd)

                # Absorb current gradient into momentum:
                beta = betas[i]
                m_hat = torch.addcmul(g, m, beta)

                denom = (m_hat.norm() * g.norm()).clamp_min(1e-8)
                cosine_sim = ((m_hat.view(-1) @ g.view(-1)) / denom).clamp(-1.0, 1.0)
                d = (1.0 - cosine_sim) * 0.5  # normalized cosine distance

                if self.adaptive:
                    m.mul_(1.0 - d).add_(g)
                    betas[i] = 1.0 - d

                else:
                    m.mul_(beta * (1.0 - d)).add_(g)

                p.sub_(m, alpha=lr)
