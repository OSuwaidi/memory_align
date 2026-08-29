"""Adaptive Memory optimizers (AM-MSGD and AM-AdamW).

Implements the stochastic Adaptive Memory algorithms from *Adaptive Memory
Momentum via a Model-Based Framework for Deep Learning Optimization*
(Topollai and Choromanska, 2025).
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class AM_MSGD(Optimizer):
    r"""Adaptive Memory Momentum SGD from Topollai and Choromanska.

    Given the current stochastic gradient :math:`g_t` and the stored EMA
    direction :math:`d_t`, AM-MSGD computes one model-wide coefficient

    .. math::

        \beta_t = \operatorname{Clip}_{[0,\beta_{\max}]}
        \frac{(1+\lambda)g_t^\top d_t
        - \langle d_t-g_t,\,g_t+\lambda d_t\rangle}
        {\lVert d_t-g_t\rVert_2^2},

    where every inner product and norm is taken over the concatenation of all
    parameters that received a gradient.  It then applies

    .. math::

        d_{t+1} = \frac{\beta_t+\lambda}{1+\lambda}d_t
                  + \frac{1-\beta_t}{1+\lambda}g_t,
        \qquad x_{t+1}=x_t-\eta d_{t+1}.

    The first direction is initialized as :math:`d_0=g_0`, as specified by the
    paper.  If :math:`d_t=g_t`, the denominator is zero and ``beta_t`` is set to
    zero by convention; the resulting direction is still exactly ``g_t``.

    ``model_lambda=0.1`` and ``beta_max=0.9`` are the fixed settings used in the
    paper's deep-learning experiments.  ``weight_decay`` is coupled L2 decay and
    is included in the gradient used both to estimate ``beta_t`` and to update
    the direction.  In keeping with the other optimizers in this repository,
    biases and one-dimensional normalization parameters are exempt from decay.

    This is the paper's EMA-style momentum recurrence.  It intentionally does
    not use PyTorch SGD's default undampened buffer ``beta * buffer + grad`` and
    does not implement a Nesterov variant.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta_max: float = 0.9,
        model_lambda: float = 0.1,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta_max < 1.0:
            raise ValueError(f"Invalid beta_max value: {beta_max}")
        if model_lambda < 0.0:
            raise ValueError(f"Invalid model_lambda value: {model_lambda}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for parameter in params:
            if not parameter.requires_grad:
                continue
            if weight_decay == 0.0 or parameter.ndim <= 1:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        if not decay_params and not no_decay_params:
            raise ValueError("Optimizer received no trainable parameters.")

        parameter_groups: list[dict[str, Any]] = []
        for group_params, group_weight_decay in ((no_decay_params, 0.0), (decay_params, weight_decay)):
            if group_params:
                parameter_groups.append({"params": group_params, "weight_decay": group_weight_decay})

        defaults = {
            "lr": lr,
            "beta_max": beta_max,
            "model_lambda": model_lambda,
        }
        super().__init__(parameter_groups, defaults)

        # Diagnostic values only; the recurrence itself is fully represented by
        # the per-parameter ``direction`` tensors in ``state``.
        self.last_beta: torch.Tensor | None = None
        self.last_effective_momentum: torch.Tensor | None = None

    @torch.no_grad()
    def step(self, closure: Callable[[], float | torch.Tensor] | None = None) -> Any:
        """Perform one AM-MSGD update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        entries: list[tuple[dict[str, Any], torch.nn.Parameter, torch.Tensor, torch.Tensor | None]] = []
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AM-MSGD does not support sparse gradients")
                if weight_decay != 0.0:
                    gradient = gradient.add(parameter, alpha=weight_decay)
                direction = self.state[parameter].get("direction")
                entries.append((group, parameter, gradient, direction))

        if not entries:
            return loss

        reference_group = entries[0][0]
        beta_max = float(reference_group["beta_max"])
        model_lambda = float(reference_group["model_lambda"])
        for group, _parameter, _gradient, _direction in entries[1:]:
            if float(group["beta_max"]) != beta_max or float(group["model_lambda"]) != model_lambda:
                raise RuntimeError("Model-wide AM-MSGD requires identical beta_max and model_lambda in every parameter group")

        initialized_entries = [entry for entry in entries if entry[3] is not None]
        if initialized_entries:
            reference_device = initialized_entries[0][2].device
            stats_dtype = torch.float64 if initialized_entries[0][2].dtype == torch.float64 else torch.float32
            numerator = torch.zeros((), device=reference_device, dtype=stats_dtype)
            denominator = torch.zeros_like(numerator)

            for _group, _parameter, gradient, direction in initialized_entries:
                assert direction is not None
                if gradient.device != reference_device:
                    raise RuntimeError("The model-wide AM-MSGD coefficient requires all parameters to be on one device")
                gradient_stats = gradient.to(dtype=stats_dtype)
                direction_stats = direction.to(dtype=stats_dtype)
                difference = direction_stats - gradient_stats
                denominator.add_(difference.square().sum())
                numerator.add_(
                    (1.0 + model_lambda) * (gradient_stats * direction_stats).sum()
                    - (difference * (gradient_stats + model_lambda * direction_stats)).sum()
                )

            safe_denominator = denominator.clamp_min(torch.finfo(stats_dtype).tiny)
            beta = torch.where(
                denominator > 0.0,
                (numerator / safe_denominator).clamp(0.0, beta_max),
                torch.zeros_like(denominator),
            )
        else:
            # Algorithm 1 initializes d_0 = g_0; there is no historical
            # direction from which to estimate beta on the first step.
            beta = torch.zeros((), device=entries[0][2].device, dtype=torch.float32)

        old_direction_weight = (beta + model_lambda) / (1.0 + model_lambda)
        gradient_weight = (1.0 - beta) / (1.0 + model_lambda)

        for group, parameter, gradient, direction in entries:
            if direction is None:
                direction = gradient.detach().clone(memory_format=torch.preserve_format)
                self.state[parameter]["direction"] = direction
            else:
                direction.mul_(old_direction_weight.to(dtype=direction.dtype))
                direction.add_(gradient * gradient_weight.to(dtype=gradient.dtype))
            parameter.add_(direction, alpha=-group["lr"])

        # Keep diagnostics on-device: a per-step ``Tensor.item()`` would force a
        # CUDA synchronization and materially distort optimizer benchmarks.
        self.last_beta = beta.detach()
        self.last_effective_momentum = old_direction_weight.detach()
        return loss


class AM_AdamW(Optimizer):
    r"""Adaptive Memory AdamW from Topollai and Choromanska.

    ``betas`` is ``(beta1_max, beta2)``: unlike AdamW, the first entry is an
    upper bound rather than a fixed first-moment coefficient.  For every
    parameter tensor and optimizer step, this implementation follows Algorithm
    3 and the stochastic/per-layer implementation details in the paper:

    .. math::

        P_t = \left(1-\beta_{1\max}\prod_{i<t}\beta_{1i}\right)
              \left(\sqrt{v_t/(1-\beta_2^t)}+\epsilon\right),

    .. math::

        d_{t+1} = (I+\lambda P_t)^{-1}
        \left[(1-\beta_{1t})g_t+(\lambda P_t+\beta_{1t}I)d_t\right],

        x_{t+1}=(1-\mu\eta_t)x_t-\eta_t P_t^{-1}d_{t+1}.

    The adaptive coefficient uses the paper's weighted metric
    :math:`P_t^{-1}(I+\lambda P_t)^{-1}` and its stochastic approximation

    .. math::

        \hat f(x_t)-f(x_t) \approx
        \eta_{t-1}g_t^\top(P_t^{-1}d_t+\mu x_t).

    The first step has no previous iterate from which to estimate that function
    gap, so it uses ``beta1_max``.  This is also the limiting initialization
    that recovers AdamW's usual bias-corrected first step when ``lambda=0``.
    A separate coefficient is computed per parameter tensor, matching the
    efficient per-layer variant used for the paper's main experiments.

    Weight decay is decoupled and, consistently with the other optimizers in
    this repository, is omitted for biases and one-dimensional normalization
    parameters.  The paper's reported setting ``lambda=0.1`` gives
    ``beta1_max = 0.9 - 0.1 * lambda = 0.89``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.89, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        model_lambda: float = 0.1,
    ) -> None:
        beta1_max, beta2 = betas
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1_max < 1.0:
            raise ValueError(f"Invalid beta1_max value: {beta1_max}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if model_lambda < 0.0:
            raise ValueError(f"Invalid model_lambda value: {model_lambda}")

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for parameter in params:
            if not parameter.requires_grad:
                continue
            if weight_decay == 0.0 or parameter.ndim <= 1:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        if not decay_params and not no_decay_params:
            raise ValueError("Optimizer received no trainable parameters.")

        parameter_groups: list[dict[str, Any]] = []
        for group_params, group_weight_decay in ((no_decay_params, 0.0), (decay_params, weight_decay)):
            if group_params:
                parameter_groups.append({"params": group_params, "weight_decay": group_weight_decay})

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "model_lambda": model_lambda,
        }
        super().__init__(parameter_groups, defaults)

        # Per-step summaries of the per-tensor coefficients. They remain on the
        # training device so logging code can decide when to synchronize.
        self.last_beta: torch.Tensor | None = None
        self.last_beta_min: torch.Tensor | None = None
        self.last_beta_max: torch.Tensor | None = None

    @torch.no_grad()
    def step(self, closure: Callable[[], float | torch.Tensor] | None = None) -> Any:
        """Perform one AM-AdamW update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        beta_total: torch.Tensor | None = None
        beta_min: torch.Tensor | None = None
        beta_max_seen: torch.Tensor | None = None
        beta_count = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1_max, beta2 = group["betas"]
            beta1_max = float(beta1_max)
            beta2 = float(beta2)
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            model_lambda = float(group["model_lambda"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AM-AdamW does not support sparse gradients")
                if torch.is_complex(parameter) or torch.is_complex(gradient):
                    raise RuntimeError("AM-AdamW does not support complex parameters")

                state = self.state[parameter]
                if not state:
                    stats_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
                    state["step"] = 0
                    state["direction"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    # Product through the previous step. Algorithm 3 bounds the
                    # unknown current factor by beta1_max in P_t.
                    state["beta_product"] = torch.ones((), device=parameter.device, dtype=stats_dtype)
                    state["previous_lr"] = lr

                state["step"] += 1
                step = int(state["step"])
                direction = state["direction"]
                exp_avg_sq = state["exp_avg_sq"]
                beta_product = state["beta_product"]

                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                stats_dtype = beta_product.dtype
                gradient_stats = gradient.to(dtype=stats_dtype)
                direction_stats = direction.to(dtype=stats_dtype)
                parameter_stats = parameter.to(dtype=stats_dtype)
                variance_stats = exp_avg_sq.to(dtype=stats_dtype)

                second_moment_correction = 1.0 - beta2**step
                first_moment_correction = 1.0 - beta1_max * beta_product
                first_moment_correction = first_moment_correction.clamp_min(torch.finfo(stats_dtype).tiny)
                preconditioner = first_moment_correction * (
                    (variance_stats / second_moment_correction).sqrt() + eps
                )

                if step == 1:
                    # The stochastic loss-gap approximation needs x_{t-1}; no
                    # such iterate exists before the first update.
                    beta = torch.full((), beta1_max, device=parameter.device, dtype=stats_dtype)
                else:
                    difference = direction_stats - gradient_stats
                    regularized_preconditioner = 1.0 + model_lambda * preconditioner
                    metric_diagonal = preconditioner.reciprocal() / regularized_preconditioner
                    denominator = (difference.square() * metric_diagonal).sum()
                    model_inner_product = (
                        difference
                        * (gradient_stats + model_lambda * preconditioner * direction_stats)
                        * metric_diagonal
                    ).sum()

                    previous_lr = float(state["previous_lr"])
                    if lr > 0.0:
                        loss_gap = previous_lr * (
                            gradient_stats
                            * (direction_stats / preconditioner + weight_decay * parameter_stats)
                        ).sum()
                        loss_gap_term = (1.0 + weight_decay * lr) * loss_gap / lr
                    else:
                        # A zero-LR scheduler endpoint has no defined 1/eta
                        # model term. The parameter update is a no-op; omitting
                        # only this term keeps state evolution finite.
                        loss_gap_term = torch.zeros((), device=parameter.device, dtype=stats_dtype)

                    weight_decay_inner_product = weight_decay * (parameter_stats * difference).sum()
                    numerator = loss_gap_term - model_inner_product - weight_decay_inner_product
                    safe_denominator = denominator.clamp_min(torch.finfo(stats_dtype).tiny)
                    beta = torch.where(
                        denominator > 0.0,
                        (numerator / safe_denominator).clamp(0.0, beta1_max),
                        torch.zeros_like(denominator),
                    )

                regularized_preconditioner = 1.0 + model_lambda * preconditioner
                direction_next = (
                    (1.0 - beta) * gradient_stats
                    + (model_lambda * preconditioner + beta) * direction_stats
                ) / regularized_preconditioner
                direction.copy_(direction_next.to(dtype=direction.dtype))

                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_((direction_next / preconditioner).to(dtype=parameter.dtype), alpha=-lr)

                beta_product.mul_(beta)
                state["previous_lr"] = lr

                beta_detached = beta.detach()
                if beta_total is None:
                    beta_total = beta_detached.clone()
                    beta_min = beta_detached.clone()
                    beta_max_seen = beta_detached.clone()
                else:
                    if beta_detached.device != beta_total.device:
                        raise RuntimeError("AM-AdamW diagnostics require all parameters to be on one device")
                    beta_total.add_(beta_detached)
                    assert beta_min is not None and beta_max_seen is not None
                    beta_min.copy_(torch.minimum(beta_min, beta_detached))
                    beta_max_seen.copy_(torch.maximum(beta_max_seen, beta_detached))
                beta_count += 1

        if beta_total is not None:
            self.last_beta = beta_total.div(beta_count)
            self.last_beta_min = beta_min
            self.last_beta_max = beta_max_seen
        return loss
