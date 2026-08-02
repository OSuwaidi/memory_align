"""Deterministic equation and shape checks for per-output MAL-SGD.

Run on CPU:
    python diagnostics/verify_per_output_mal.py --device cpu

Run on Apple Silicon outside a sandbox that denies Metal access:
    python diagnostics/verify_per_output_mal.py --device mps
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mal_sgd import MAL_SGD


ATOL = 2e-6
RTOL = 2e-6


def parameter_group_and_index(
    optimizer: MAL_SGD,
    parameter: torch.nn.Parameter,
) -> tuple[dict, int]:
    for group in optimizer.param_groups:
        for index, candidate in enumerate(group["params"]):
            if candidate is parameter:
                return group, index
    raise AssertionError("parameter is missing from optimizer groups")


def adaptive_beta(
    optimizer: MAL_SGD,
    parameter: torch.nn.Parameter,
) -> torch.Tensor:
    group, index = parameter_group_and_index(optimizer, parameter)
    return group["beta"][index]


def reference_cosine(
    candidate: torch.Tensor,
    gradient: torch.Tensor,
    *,
    per_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if per_output and gradient.ndim > 1:
        candidate_rows = candidate.reshape(candidate.shape[0], -1)
        gradient_rows = gradient.reshape(gradient.shape[0], -1)
        denominator = (candidate_rows.norm(dim=1) * gradient_rows.norm(dim=1)).clamp_min(1e-8)
        cosine = ((candidate_rows * gradient_rows).sum(dim=1) / denominator).clamp(-1.0, 1.0)
        output_shape = (gradient.shape[0],) + (1,) * (gradient.ndim - 1)
        return cosine.reshape(output_shape), gradient_rows.norm(dim=1).reshape(output_shape) > 0.0

    candidate_vector = candidate.reshape(-1)
    gradient_vector = gradient.reshape(-1)
    denominator = (candidate_vector.norm() * gradient_vector.norm()).clamp_min(1e-8)
    cosine = ((candidate_vector * gradient_vector).sum() / denominator).clamp(
        -1.0,
        1.0,
    )
    return cosine, gradient_vector.norm() > 0.0


def reference_step(
    parameter: torch.Tensor,
    momentum: torch.Tensor,
    gradient: torch.Tensor,
    beta_probe: torch.Tensor | float,
    *,
    lr: float,
    weight_decay: float,
    adaptive: bool,
    adaptation_rate: float,
    nesterov: bool,
    per_output: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    effective_gradient = gradient + weight_decay * parameter
    candidate = effective_gradient + beta_probe * momentum
    cosine, has_gradient = reference_cosine(
        candidate,
        effective_gradient,
        per_output=per_output,
    )
    retention = (1.0 + cosine) * 0.5

    if adaptive:
        assert isinstance(beta_probe, torch.Tensor)
        target = torch.where(has_gradient, retention, beta_probe)
        coefficient = torch.lerp(beta_probe, target, adaptation_rate)
    else:
        assert isinstance(beta_probe, float)
        aligned = beta_probe * retention
        coefficient = torch.where(
            has_gradient,
            aligned,
            torch.full_like(aligned, beta_probe),
        )

    next_momentum = coefficient * momentum + effective_gradient
    if nesterov:
        next_parameter = parameter - lr * (effective_gradient + coefficient * next_momentum)
    else:
        next_parameter = parameter - lr * next_momentum
    return next_parameter, next_momentum, coefficient


def check_shape_contract(device: torch.device) -> None:
    cases = {
        (7, 11): (7, 1),
        (7, 11, 5): (7, 1, 1),
        (16, 8, 3, 3): (16, 1, 1, 1),
        (7, 3, 2, 3, 4): (7, 1, 1, 1, 1),
        (17,): (),
        (): (),
    }
    for parameter_shape, expected_beta_shape in cases.items():
        parameter = torch.nn.Parameter(torch.randn(parameter_shape, device=device))
        optimizer = MAL_SGD(
            [parameter],
            beta=0.73,
            adaptive=True,
            per_output=True,
        )
        assert adaptive_beta(optimizer, parameter).shape == expected_beta_shape

        tensor_optimizer = MAL_SGD(
            [torch.nn.Parameter(parameter.detach().clone())],
            beta=0.73,
            adaptive=True,
            per_output=False,
        )
        assert tensor_optimizer.param_groups[0]["beta"][0].shape == ()


def check_equations(
    device: torch.device,
    *,
    per_output: bool,
    adaptive: bool,
    nesterov: bool,
    weight_decay: float,
) -> float:
    generator = torch.Generator(device=device).manual_seed(1401)
    initial_parameter = torch.randn(
        16,
        8,
        3,
        3,
        generator=generator,
        device=device,
    )
    initial_momentum = torch.randn(
        initial_parameter.shape,
        generator=generator,
        device=device,
    )
    gradient = torch.randn(
        initial_parameter.shape,
        generator=generator,
        device=device,
    )
    # A block with no fresh direction must retain its prior coefficient.
    gradient[3].zero_()

    parameter = torch.nn.Parameter(initial_parameter.clone())
    optimizer = MAL_SGD(
        [parameter],
        lr=0.037,
        beta=0.73,
        weight_decay=weight_decay,
        adaptive=adaptive,
        c=0.31,
        nesterov=nesterov,
        per_output=per_output,
    )
    group, index = parameter_group_and_index(optimizer, parameter)
    group["momentum"][index].copy_(initial_momentum)
    beta_probe: torch.Tensor | float
    if adaptive:
        beta_probe = group["beta"][index].clone()
    else:
        beta_probe = group["beta"]

    expected_parameter, expected_momentum, expected_beta = reference_step(
        initial_parameter,
        initial_momentum,
        gradient,
        beta_probe,
        lr=0.037,
        weight_decay=weight_decay,
        adaptive=adaptive,
        adaptation_rate=0.31,
        nesterov=nesterov,
        per_output=per_output,
    )

    parameter.grad = gradient.clone()
    optimizer.step()

    torch.testing.assert_close(
        parameter,
        expected_parameter,
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        group["momentum"][index],
        expected_momentum,
        atol=ATOL,
        rtol=RTOL,
    )
    if adaptive:
        torch.testing.assert_close(
            group["beta"][index],
            expected_beta,
            atol=ATOL,
            rtol=RTOL,
        )

    max_errors = [
        (parameter - expected_parameter).abs().max(),
        (group["momentum"][index] - expected_momentum).abs().max(),
    ]
    if adaptive:
        max_errors.append((group["beta"][index] - expected_beta).abs().max())
    return max(float(value.detach().cpu()) for value in max_errors)


def check_independent_outputs(device: torch.device) -> None:
    parameter = torch.nn.Parameter(torch.zeros(4, 2, 1, 1, device=device))
    optimizer = MAL_SGD(
        [parameter],
        lr=0.0,
        beta=0.8,
        adaptive=True,
        c=1.0,
        per_output=True,
    )
    group, index = parameter_group_and_index(optimizer, parameter)
    momentum = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[-2.0]], [[0.0]]],
            [[[0.0]], [[2.0]]],
            [[[3.0]], [[4.0]]],
        ],
        device=device,
    )
    gradient = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[1.0]], [[0.0]]],
            [[[2.0]], [[0.0]]],
            [[[0.0]], [[0.0]]],
        ],
        device=device,
    )
    group["momentum"][index].copy_(momentum)
    parameter.grad = gradient

    beta_before = adaptive_beta(optimizer, parameter).clone()
    candidate = gradient + beta_before * momentum
    expected_cosine, has_gradient = reference_cosine(
        candidate,
        gradient,
        per_output=True,
    )
    expected_beta = torch.where(
        has_gradient,
        (1.0 + expected_cosine) * 0.5,
        beta_before,
    )
    optimizer.step()

    actual_beta = adaptive_beta(optimizer, parameter)
    assert actual_beta.shape == (4, 1, 1, 1)
    torch.testing.assert_close(actual_beta, expected_beta, atol=ATOL, rtol=RTOL)
    assert torch.unique(actual_beta).numel() >= 3
    torch.testing.assert_close(actual_beta[3], beta_before[3])


def check_state_round_trip(device: torch.device) -> None:
    source = torch.nn.Parameter(torch.randn(5, 3, 3, 3, device=device))
    source_optimizer = MAL_SGD(
        [source],
        lr=0.02,
        beta=0.81,
        adaptive=True,
        per_output=True,
    )
    source.grad = torch.randn_like(source)
    source_optimizer.step()
    saved = source_optimizer.state_dict()

    destination = torch.nn.Parameter(source.detach().clone())
    destination_optimizer = MAL_SGD(
        [destination],
        lr=0.5,
        beta=0.2,
        adaptive=True,
        per_output=True,
    )
    destination_optimizer.load_state_dict(saved)

    source_group, source_index = parameter_group_and_index(source_optimizer, source)
    destination_group, destination_index = parameter_group_and_index(
        destination_optimizer,
        destination,
    )
    torch.testing.assert_close(
        source_group["momentum"][source_index],
        destination_group["momentum"][destination_index],
    )
    torch.testing.assert_close(
        source_group["beta"][source_index],
        destination_group["beta"][destination_index],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was explicitly requested but torch.backends.mps.is_available() is false; refusing to silently substitute CPU.")

    device = torch.device(args.device)
    check_shape_contract(device)
    check_independent_outputs(device)
    check_state_round_trip(device)

    errors = {}
    for per_output in (False, True):
        for adaptive in (False, True):
            for nesterov in (False, True):
                for weight_decay in (0.0, 5e-4):
                    name = f"scope={'output' if per_output else 'tensor'};adaptive={adaptive};nesterov={nesterov};wd={weight_decay}"
                    errors[name] = check_equations(
                        device,
                        per_output=per_output,
                        adaptive=adaptive,
                        nesterov=nesterov,
                        weight_decay=weight_decay,
                    )

    result = {
        "status": "pass",
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "equation_cases": len(errors),
        "maximum_absolute_error": max(errors.values()),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
