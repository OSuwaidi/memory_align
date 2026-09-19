"""Focused executable checks for Lion and AGAM-Lion."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from optims.lion_opt import AGAM_Lion, Lion


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


class LionOptimizerChecks(unittest.TestCase):
    def test_lion_matches_reference_update(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        optimizer = Lion([parameter], lr=0.1, betas=(0.9, 0.99), weight_decay=0.2)
        gradient = torch.tensor([2.0, -3.0], dtype=torch.float64)

        parameter.grad = gradient.clone()
        optimizer.step()

        expected_parameter = torch.tensor([1.0, -2.0], dtype=torch.float64) * 0.98 - 0.1 * gradient.sign()
        assert_close(parameter, expected_parameter)
        assert_close(optimizer.state[parameter]["exp_avg"], 0.01 * gradient)

    def test_agam_preserves_base_lion_persistent_state(self) -> None:
        initial = torch.tensor([0.4, -0.7, 1.2], dtype=torch.float64)
        lion_parameter = torch.nn.Parameter(initial.clone())
        agam_parameter = torch.nn.Parameter(initial.clone())
        lion = Lion([lion_parameter], lr=0.03, betas=(0.9, 0.99))
        agam = AGAM_Lion([agam_parameter], lr=0.03, betas=(0.9, 0.99))

        gradients = (
            torch.tensor([1.0, -2.0, 0.4], dtype=torch.float64),
            torch.tensor([-0.8, 0.3, -1.5], dtype=torch.float64),
            torch.tensor([0.2, 0.9, -0.1], dtype=torch.float64),
        )
        for gradient in gradients:
            lion_parameter.grad = gradient.clone()
            agam_parameter.grad = gradient.clone()
            lion.step()
            agam.step()
            assert_close(lion.state[lion_parameter]["exp_avg"], agam.state[agam_parameter]["exp_avg"])

    def test_conflicting_gradient_rejects_immediate_memory(self) -> None:
        lion_parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        agam_parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        lion = Lion([lion_parameter], lr=0.1, betas=(0.9, 0.99))
        agam = AGAM_Lion([agam_parameter], lr=0.1, betas=(0.9, 0.99))
        lion.state[lion_parameter]["exp_avg"] = torch.tensor([1.0], dtype=torch.float64)
        agam.state[agam_parameter]["exp_avg"] = torch.tensor([1.0], dtype=torch.float64)
        conflicting_gradient = torch.tensor([-0.2], dtype=torch.float64)

        lion_parameter.grad = conflicting_gradient.clone()
        agam_parameter.grad = conflicting_gradient.clone()
        lion.step()
        agam.step()

        # Base Lion follows stale positive memory; AGAM's scalar gate is zero
        # in this exact 1-D conflict and therefore follows the fresh gradient.
        assert_close(lion_parameter, torch.tensor([-0.1], dtype=torch.float64))
        assert_close(agam_parameter, torch.tensor([0.1], dtype=torch.float64))
        assert_close(lion.state[lion_parameter]["exp_avg"], agam.state[agam_parameter]["exp_avg"])

    def test_perfect_alignment_and_undefined_alignment_recover_lion(self) -> None:
        for gradient in (
            torch.tensor([0.3, 0.3], dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
        ):
            with self.subTest(gradient=gradient.tolist()):
                initial = torch.tensor([1.0, -1.0], dtype=torch.float64)
                lion_parameter = torch.nn.Parameter(initial.clone())
                agam_parameter = torch.nn.Parameter(initial.clone())
                lion = Lion([lion_parameter], lr=0.1, betas=(0.9, 0.99))
                agam = AGAM_Lion([agam_parameter], lr=0.1, betas=(0.9, 0.99))
                memory = torch.tensor([0.5, 0.7], dtype=torch.float64)
                lion.state[lion_parameter]["exp_avg"] = memory.clone()
                agam.state[agam_parameter]["exp_avg"] = memory.clone()

                lion_parameter.grad = gradient.clone()
                agam_parameter.grad = gradient.clone()
                lion.step()
                agam.step()
                assert_close(agam_parameter, lion_parameter)
                assert_close(agam.state[agam_parameter]["exp_avg"], lion.state[lion_parameter]["exp_avg"])

    def test_state_dict_round_trip(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        optimizer = AGAM_Lion([parameter], lr=0.03, betas=(0.9, 0.99), pwr=1.0)
        parameter.grad = torch.tensor([0.4, -0.2], dtype=torch.float64)
        optimizer.step()
        checkpoint = copy.deepcopy(optimizer.state_dict())

        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = AGAM_Lion([restored_parameter], lr=0.03, betas=(0.9, 0.99), pwr=1.0)
        restored.load_state_dict(checkpoint)
        final_gradient = torch.tensor([-0.3, 0.8], dtype=torch.float64)
        parameter.grad = final_gradient.clone()
        restored_parameter.grad = final_gradient.clone()
        optimizer.step()
        restored.step()

        assert_close(restored_parameter, parameter)
        assert_close(restored.state[restored_parameter]["exp_avg"], optimizer.state[parameter]["exp_avg"])


if __name__ == "__main__":
    unittest.main()
