"""Focused numerical checks for the QHM-style MAL-SGDM recurrence."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from optims.mal_opt import MAL_SGDM


class MALSGDMQHMChecks(unittest.TestCase):
    def test_transient_complement_estimators_match_their_equations(self) -> None:
        beta, lr = 0.8, 0.05
        gradients = (
            torch.tensor([1.0, -2.0], dtype=torch.float64),
            torch.tensor([-0.4, 0.7], dtype=torch.float64),
        )

        for unbias in ("none", "buffer", "estimator"):
            with self.subTest(unbias=unbias):
                parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2], dtype=torch.float64))
                optimizer = MAL_SGDM(
                    [parameter],
                    lr=lr,
                    beta=beta,
                    gate_mode="attenuate",
                    gradient_weight_mode="complement",
                    unbias=unbias,
                )

                parameter.grad = gradients[0].clone()
                before_first = parameter.detach().clone()
                optimizer.step()
                first_update = (1.0 - beta) * gradients[0] if unbias == "none" else gradients[0]
                torch.testing.assert_close(parameter, before_first - lr * first_update)

                old_memory = optimizer.state[parameter]["momentum_buffer"].clone()
                before_second = parameter.detach().clone()
                gradient = gradients[1]
                probe = beta * old_memory + (1.0 - beta) * gradient
                cosine = torch.dot(gradient, probe) / (
                    torch.linalg.vector_norm(gradient) * torch.linalg.vector_norm(probe)
                )
                gate = (1.0 + cosine.clamp(-1.0, 1.0)) * 0.5
                memory_weight = beta * gate
                effective = memory_weight * old_memory + (1.0 - memory_weight) * gradient

                if unbias == "none":
                    expected_update = effective
                elif unbias == "buffer":
                    expected_update = (
                        effective + (gate - 1.0) * beta**2 * gradient
                    ) / (1.0 - beta**2)
                else:
                    expected_update = effective / (1.0 - memory_weight * beta)

                parameter.grad = gradient.clone()
                optimizer.step()
                torch.testing.assert_close(parameter, before_second - lr * expected_update)
                torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], probe)
                self.assertEqual(optimizer.state[parameter]["step"], 2)

    def test_estimator_zero_gradient_uses_the_effective_fallback_coefficient(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
        optimizer = MAL_SGDM(
            [parameter],
            lr=0.1,
            beta=0.8,
            gradient_weight_mode="complement",
            unbias="estimator",
        )
        parameter.grad = torch.tensor([2.0, -3.0], dtype=torch.float64)
        optimizer.step()

        before = parameter.detach().clone()
        old_memory = optimizer.state[parameter]["momentum_buffer"].clone()
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        expected = 0.8 * old_memory / (1.0 - 0.8**2)
        torch.testing.assert_close(parameter, before - 0.1 * expected)

    def test_default_fixed_mode_retains_heavy_ball_recurrence(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([0.5, -0.5], dtype=torch.float64))
        optimizer = MAL_SGDM([parameter], lr=0.1, beta=0.9)
        for gradient in (
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([-0.5, 0.25], dtype=torch.float64),
        ):
            old_memory = optimizer.state[parameter].get("momentum_buffer", torch.zeros_like(parameter)).clone()
            probe = gradient + 0.9 * old_memory
            cosine = torch.dot(gradient, probe) / (
                torch.linalg.vector_norm(gradient) * torch.linalg.vector_norm(probe)
            )
            coefficient = 0.9 * (1.0 + cosine.clamp(-1.0, 1.0)) * 0.5
            expected_update = gradient + coefficient * old_memory
            before = parameter.detach().clone()
            parameter.grad = gradient.clone()
            optimizer.step()
            torch.testing.assert_close(parameter, before - 0.1 * expected_update)
            torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], probe)

    def test_invalid_or_ambiguous_combinations_are_rejected(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(2))
        invalid = (
            {"gradient_weight_mode": "complement", "gate_mode": "replace"},
            {"gradient_weight_mode": "complement", "nesterov": True},
            {"gradient_weight_mode": "complement", "unbias": "buffer", "in_place": True},
            {"gradient_weight_mode": "complement", "unbias": "estimator", "scale": True},
            {"gradient_weight_mode": "fixed", "unbias": "buffer"},
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                MAL_SGDM([parameter], **options)

    def test_checkpoint_round_trip_preserves_new_state(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        optimizer = MAL_SGDM(
            [parameter],
            beta=0.7,
            gradient_weight_mode="complement",
            unbias="buffer",
        )
        parameter.grad = torch.tensor([0.2, -0.6], dtype=torch.float64)
        optimizer.step()
        checkpoint = copy.deepcopy(optimizer.state_dict())

        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = MAL_SGDM(
            [restored_parameter],
            beta=0.7,
            gradient_weight_mode="complement",
            unbias="buffer",
        )
        restored.load_state_dict(checkpoint)
        next_gradient = torch.tensor([-0.3, 0.1], dtype=torch.float64)
        parameter.grad = next_gradient.clone()
        restored_parameter.grad = next_gradient.clone()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(restored_parameter, parameter)
        self.assertEqual(restored.state[restored_parameter]["step"], 2)


if __name__ == "__main__":
    unittest.main()
