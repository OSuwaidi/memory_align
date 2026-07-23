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
        adaptive: bool = True,
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
                    "beta": [p.new_tensor(0.0) for p in no_decay_params] if adaptive else beta,
                }
            )
        if decay_params:
            optim_groups.append(
                {
                    "params": decay_params,
                    "momentum": decay_momentum,
                    "weight_decay": weight_decay,
                    "beta": [p.new_tensor(0.0) for p in decay_params] if adaptive else beta,
                },
            )

        defaults = dict(
            lr=lr, nesterov=nesterov
        )  # shared across all optim/param groups
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
        vec: torch.Tensor,
        params: Iterable[torch.nn.Parameter],
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
            nesterov = group["nesterov"]

            for i, (p, m) in enumerate(zip(group["params"], group["momentum"])):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    g.add_(p, alpha=wd)

                # Absorb current gradient into momentum:
                if self.adaptive:
                    beta = betas[i]  # per-param 0-dim tensor
                    m_hat = torch.addcmul(g, m, beta)
                else:
                    m_hat = torch.add(g, m, alpha=betas)  # group-level float scalar

                if g.ndim > 1:
                    # One alignment per output unit: Linear (out, in) reduces dim 1 (each
                    # neuron's incoming weights together, no cross-neuron grouping);
                    # ConvNd (out, in, k, ...) reduces dims (1..N-1), i.e. each output
                    # kernel's flattened (in, k, k) block. c broadcasts as (out, 1, ..., 1).
                    # NOTE: Tensor.norm(dim=<3-tuple>) dispatches to matrix_norm and
                    # raises; vector_norm is the correct block-norm over arbitrary dims.
                    dims = tuple(range(1, g.ndim))
                    denom = (
                        torch.linalg.vector_norm(m_hat, dim=dims, keepdim=True)
                        * torch.linalg.vector_norm(g, dim=dims, keepdim=True)
                    ).clamp_min(1e-8)
                    cosine_sim = ((m_hat * g).sum(dims, keepdim=True) / denom).clamp(-1.0, 1.0)
                else:
                    # 1-D params (biases, norm affines) are treated holistically: their only
                    # "per-axis" reading would be per-coordinate sign gating, which destroys
                    # the memory; whole-tensor gating of these also tested better
                    denom = (m_hat.norm() * g.norm()).clamp_min(1e-8)
                    cosine_sim = ((m_hat.view(-1) @ g.view(-1)) / denom).clamp(-1.0, 1.0)

                d = (1.0 - cosine_sim) * 0.5  # normalized cosine distance
                c = 1.0 - d  # per-output-unit memory-aligned retention

                # Effective momentum coefficient for this step
                if self.adaptive:
                    betas[i] = c
                else:
                    c *= betas

                m.mul_(c).add_(g)

                if nesterov:
                    # NAG look-ahead with the SAME aligned coefficient that decayed the
                    # memory: u = g + c*m, so perfect alignment (d=0) recovers vanilla
                    # Nesterov exactly, while a suspect memory gets less look-ahead
                    p.sub_(g, alpha=lr)
                    p.addcmul_(m, c, value=-lr)
                else:
                    p.sub_(m, alpha=lr)


class MAL_ADAMW(Optimizer):
    """Memory-ALigned AdamW for transformer training (ViT / LLM).

    The alignment gate modulates ONLY the first moment's EMA coefficient, per layer.
    Probe: the candidate EMA m_hat = beta1*m + (1-beta1)*g vs the fresh gradient;
    effective coefficient c = 1-d (adaptive, stored per layer) or beta1*(1-d) (static).
    The EMA form m <- c*m + (1-c)*g keeps unit mass under a dynamic c, so the update-
    magnitude concern from the SGDM variant does not arise. The second moment (scale
    tracker, beta2) and its bias correction are standard AdamW and are never gated.

    First-moment bias correction is EXACT under dynamic per-layer c: the running
    product bc_prod = prod_s c_s is tracked per parameter and the correction is
    1 - bc_prod (reduces to 1 - beta1^t for constant c). Adaptive c is capped at
    1 - 1e-4 because c = 1 is the one degenerate EMA value (zero mass on g freezes
    the memory and zeroes the correction); with the cap, a perfectly-aligned first
    step reproduces plain AdamW's first step exactly.

    Weight decay is decoupled (AdamW) and never enters the alignment signal.
    LayerNorm gains and biases (ndim <= 1) keep a fixed beta1 unless align_1d=True:
    cosine similarity on tiny vectors is noise-dominated.
    """

    MAX_BETA1 = 1.0 - 1e-4

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        adaptive: bool = True,
        align_1d: bool = False,
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

        beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.adaptive = adaptive
        self.align_1d = align_1d
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

        for group_params, group_wd in (
            (no_decay_params, 0.0),
            (decay_params, weight_decay),
        ):
            if group_params:
                optim_groups.append(
                    {
                        "params": group_params,
                        "m": [torch.zeros_like(p) for p in group_params],
                        "v": [torch.zeros_like(p) for p in group_params],
                        "weight_decay": group_wd,
                        "beta": [p.new_tensor(beta1) for p in group_params],
                        "bc_prod": [p.new_tensor(1.0) for p in group_params],
                    }
                )

        defaults = dict(lr=lr)  # shared across all optim/param groups
        super().__init__(optim_groups, defaults)

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc2_sqrt = (1.0 - self.beta2**self.t) ** 0.5

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            betas = group["beta"]
            bc_prods = group["bc_prod"]

            for i, (p, m, v) in enumerate(zip(group["params"], group["m"], group["v"])):
                if p.grad is None:
                    continue

                g = p.grad
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled decay; never enters the gate

                beta1 = betas[i]
                if p.ndim > 1 or self.align_1d:
                    # Alignment of the candidate EMA with the fresh gradient:
                    m_hat = torch.lerp(g, m, beta1)  # = beta1*m + (1-beta1)*g
                    denom = (m_hat.norm() * g.norm()).clamp_min(1e-8)
                    cosine_sim = ((m_hat.view(-1) @ g.view(-1)) / denom).clamp(
                        -1.0, 1.0
                    )
                    d = (1.0 - cosine_sim) * 0.5  # normalized cosine distance

                    if self.adaptive:
                        c = (1.0 - d).clamp_max(self.MAX_BETA1)
                        betas[i] = c
                    else:
                        c = beta1 * (1.0 - d)
                else:
                    c = beta1  # tiny 1-D params: plain EMA

                m.lerp_(g, 1.0 - c)  # m <- c*m + (1-c)*g
                bc_prod = bc_prods[i] * c
                bc_prods[i] = bc_prod
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                # AdamW update with exact first-moment bias correction under dynamic c
                u = m.div(v.sqrt().div_(bc2_sqrt).add_(self.eps))
                u.div_(1.0 - bc_prod)
                p.add_(u, alpha=-lr)
