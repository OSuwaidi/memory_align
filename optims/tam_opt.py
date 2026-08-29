from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer


class _TorqueAwareOptimizer(Optimizer):
    """Shared model-wide torque computation used by TAM and AdaTAM."""

    def _init_torque_state(self) -> None:
        params = [p for group in self.param_groups for p in group["params"]]
        if not params:
            raise ValueError("Optimizer received no trainable parameters.")

        self._torque_anchor = params[0]
        self.state[self._torque_anchor]["s_hat"] = torch.zeros((), device=self._torque_anchor.device, dtype=torch.float32)

    def _torque_scale(
        self,
        moment_and_grad: list[tuple[torch.Tensor, torch.Tensor]],
        gamma: float,
        torque_eps: float,
    ) -> torch.Tensor:
        """Return eps + (1 + smoothed cosine(momentum, gradient)) / 2."""
        s_hat = self.state[self._torque_anchor]["s_hat"]
        dot = torch.zeros_like(s_hat)
        momentum_sq = torch.zeros_like(s_hat)
        gradient_sq = torch.zeros_like(s_hat)

        for momentum, grad in moment_and_grad:
            if grad.is_sparse:
                raise RuntimeError("TAM optimizers do not support sparse gradients")
            if grad.device != s_hat.device:
                raise RuntimeError("The model-wide TAM correlation requires parameters on one device")

            m = momentum.to(dtype=s_hat.dtype)
            g = grad.to(dtype=s_hat.dtype)
            dot.add_((m * g).sum())
            momentum_sq.add_((m * m).sum())
            gradient_sq.add_((g * g).sum())

        # A zero previous momentum (including the first step) has zero alignment.
        denominator = momentum_sq.sqrt() * gradient_sq.sqrt()
        cosine = torch.where(
            denominator > 0.0,
            dot / denominator.clamp_min(torch.finfo(s_hat.dtype).tiny),
            torch.zeros_like(dot),
        ).clamp_(-1.0, 1.0)

        s_hat.mul_(gamma).add_(cosine, alpha=1.0 - gamma)
        return s_hat.add(1.0).mul_(0.5).add_(torque_eps)


class TAM_SGDM(_TorqueAwareOptimizer):
    """Torque-Aware Momentum (TAM) applied to stochastic gradient descent.

    The implementation follows the paper's heavy-ball recurrence directly:

        m_t = beta * m_(t-1) + (torque_eps + d_t) * g_t
        theta_t = theta_(t-1) - lr * m_t

    where d_t is derived from a model-wide, exponentially smoothed cosine
    similarity between the previous momentum and current gradient.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta: float = 0.9,
        gamma: float = 0.9,
        torque_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if torque_eps <= 0.0:
            raise ValueError(f"Invalid torque_eps value: {torque_eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

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

        defaults = {"lr": lr, "beta": beta, "gamma": gamma, "torque_eps": torque_eps}
        super().__init__(optim_groups, defaults)
        self._init_torque_state()

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        entries: list[tuple[dict, torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []
        for group in self.param_groups:
            wd = group["weight_decay"]
            for p, momentum in zip(group["params"], group["momentum"]):
                if p.grad is None:
                    continue
                grad = p.grad if wd == 0.0 else p.grad.add(p, alpha=wd)
                entries.append((group, p, momentum, grad))

        if not entries:
            return loss

        first_group = self.param_groups[0]
        torque_scale = self._torque_scale(
            [(momentum, grad) for _, _, momentum, grad in entries],
            first_group["gamma"],
            first_group["torque_eps"],
        )

        for group, p, momentum, grad in entries:
            momentum.mul_(group["beta"]).add_(grad * torque_scale.to(dtype=grad.dtype))
            p.add_(momentum, alpha=-group["lr"])

        return loss


class AdaTAMW(_TorqueAwareOptimizer):
    """AdaTAM with AdamW-style decoupled weight decay.

    This intentionally implements the AdaTAM recurrence stated in the paper:
    its TAM first moment is not an EMA, and neither moment is bias-corrected.
    Setting ``weight_decay=0`` gives AdaTAM exactly.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        gamma: float = 0.9,
        torque_eps: float = 1e-8,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if torque_eps <= 0.0:
            raise ValueError(f"Invalid torque_eps value: {torque_eps}")
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
            if weight_decay == 0.0 or p.ndim <= 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

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

        defaults = {
            "lr": lr,
            "betas": betas,
            "gamma": gamma,
            "torque_eps": torque_eps,
            "eps": eps,
        }
        super().__init__(optim_groups, defaults)
        self._init_torque_state()

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        entries: list[tuple[dict, torch.nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for group in self.param_groups:
            for p, momentum, variance in zip(group["params"], group["m"], group["v"]):
                if p.grad is None:
                    continue
                entries.append((group, p, momentum, variance, p.grad))

        if not entries:
            return loss

        first_group = self.param_groups[0]
        torque_scale = self._torque_scale(
            [(momentum, grad) for _, _, momentum, _, grad in entries],
            first_group["gamma"],
            first_group["torque_eps"],
        )

        for group, p, momentum, variance, grad in entries:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]

            if wd > 0.0:
                p.mul_(1.0 - lr * wd)

            momentum.mul_(beta1).add_(grad * torque_scale.to(dtype=grad.dtype))
            variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            p.addcdiv_(momentum, variance.sqrt().add_(group["eps"]), value=-lr)

        return loss
