import warnings
from typing import Iterable

import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        EMA: bool = True,
        couple: bool = False,
        per: bool = True,
        mem_align: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.EMA = EMA
        self.couple = couple
        self.per = per
        self.mem_align = mem_align

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
            if per:
                t = [0] * len(no_decay_params)
            else:
                t = 0
                no_decay_momentum = self._params_to_vec(no_decay_momentum)
            optim_groups.append(
                {
                    "params": no_decay_params,
                    "momentum": no_decay_momentum,
                    "t": t,
                    "weight_decay": 0.0,
                }
            )
        if decay_params:
            if per:
                t = [0] * len(decay_params)
            else:
                t = 0
                decay_momentum = self._params_to_vec(decay_momentum)
            optim_groups.append(
                {
                    "params": decay_params,
                    "momentum": decay_momentum,
                    "t": t,
                    "weight_decay": weight_decay,
                },
            )

        defaults = dict(lr=lr, beta=beta)  # shared across all optim/param groups
        super().__init__(
            optim_groups, defaults
        )  # exposes "self.param_groups" attribute

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
        vec: torch.Tensor, params: Iterable[torch.nn.Parameter]
    ) -> None:
        pointer = 0
        for param in params:
            end = pointer + param.numel()
            param.copy_(vec[pointer:end].view_as(param))
            pointer = end

    @staticmethod
    def bounce(G1: torch.Tensor, G2: torch.Tensor, tau: float = 0.0) -> torch.Tensor:
        # "Global" per optimizer param group (not truly model-global)
        return (G1 @ G2) < (-tau * G1.norm() * G2.norm())

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["beta"]

            if self.per:
                ts: list[int] = group["t"]
                for i, (p, m) in enumerate(zip(group["params"], group["momentum"])):
                    g = p.grad if p.grad is not None else torch.zeros_like(p)
                    ts[i] += 1

                    if wd > 0.0:
                        if self.couple:
                            g.add_(p, alpha=wd)
                        else:
                            p.mul_(1.0 - lr * wd)

                    if self.EMA:
                        m.lerp_(g, weight=1.0 - beta)
                    else:
                        m.mul_(beta).add_(g)

                    if self.mem_align:
                        bounce_cond = self.bounce(m.view(-1), g.view(-1))
                        if bounce_cond:
                            ts[i] = 1
                            m.zero_()
                            if self.EMA:
                                m.add_(g, alpha=1.0 - beta)
                            else:
                                m.add_(g)

                    unbias_m = m / (1.0 - beta ** ts[i]) if self.EMA else m
                    p.sub_(unbias_m, alpha=lr)

            else:
                params = group["params"]
                M = group["momentum"]
                P = self._params_to_vec(params)
                G = self._param_grads_to_vec(params)
                group["t"] += 1

                if wd > 0.0:
                    if self.couple:
                        G.add_(P, alpha=wd)
                    else:
                        P.mul_(1.0 - lr * wd)

                if self.EMA:
                    M.lerp_(G, weight=1.0 - beta)
                else:
                    M.mul_(beta).add_(G)

                if self.mem_align:
                    bounce_cond = self.bounce(M, G)
                    if bounce_cond:
                        group["t"] = 1
                        M.zero_()
                        if self.EMA:
                            M.add_(G, alpha=1.0 - beta)
                        else:
                            M.add_(G)

                t = group["t"]
                unbias_m = M / (1.0 - beta**t) if self.EMA else M
                P.sub_(unbias_m, alpha=lr)
                self._assign_vec_to_params(P, params)
