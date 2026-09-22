"""Focused checks for AGAM-AdamW first-moment correction modes."""

from __future__ import annotations

import argparse
import copy
import sys
import unittest
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from optims.agam_opt import AGAM_AdamW
from sweeps.agam_adamw_mae_bias_correction_sweep import EXPECTED_RUNS, build_sweep


class AGAMAdamWBiasCorrectionChecks(unittest.TestCase):
    def test_sweep_is_exactly_three_paired_runs(self) -> None:
        sweep = build_sweep(
            argparse.Namespace(
                program="tasks/mae_pretrain.py",
                sweep_name="check",
                source_revision="test",
                data_dir="/tmp/tiny-imagenet-200",
                output_dir="/tmp/output",
            )
        )
        cardinality = 1
        for parameter in sweep["parameters"].values():
            cardinality *= len(parameter["values"])
        self.assertEqual(cardinality, EXPECTED_RUNS)
        self.assertEqual(
            sweep["parameters"]["MAL_config"]["values"],
            ("False,1.0,none,attenuate,update,complement",),
        )
        self.assertEqual(sweep["parameters"]["agam_first_moment_correction"]["values"], ("standard",))
        self.assertEqual(sweep["parameters"]["base_lr"]["values"], (1.5e-3,))
        self.assertEqual(sweep["parameters"]["weight_decay"]["values"], (5e-2,))
        self.assertEqual(sweep["parameters"]["seed"]["values"], (42, 1337, 2026))

    def test_standard_correction_uses_one_minus_beta_power_t(self) -> None:
        beta1, beta2, eps, lr = 0.8, 0.9, 1e-12, 0.05
        first_gradient = torch.tensor([1.0, -2.0], dtype=torch.float64)
        second_gradient = torch.tensor([0.5, 0.4], dtype=torch.float64)
        parameter = torch.nn.Parameter(torch.tensor([0.7, -0.3], dtype=torch.float64))
        optimizer = AGAM_AdamW(
            [parameter],
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            pwr=1.0,
            align="moment",
            in_place=False,
            scale="none",
            gate_mode="attenuate",
            gradient_weight_mode="complement",
            first_moment_correction="standard",
        )

        parameter.grad = first_gradient.clone()
        optimizer.step()
        after_first = parameter.detach().clone()
        state = optimizer.state[parameter]
        old_m = state["exp_avg"].clone()
        old_v = state["exp_avg_sq"].clone()

        m_probe = beta1 * old_m + (1.0 - beta1) * second_gradient
        cosine = torch.dot(second_gradient, m_probe) / (
            torch.linalg.vector_norm(second_gradient) * torch.linalg.vector_norm(m_probe)
        )
        memory_weight = beta1 * (1.0 + cosine.clamp(-1.0, 1.0)) * 0.5
        m_effective = memory_weight * old_m + (1.0 - memory_weight) * second_gradient
        v_expected = beta2 * old_v + (1.0 - beta2) * second_gradient.square()
        denominator = (v_expected / (1.0 - beta2**2)).sqrt() + eps
        expected_update = (m_effective / (1.0 - beta1**2)) / denominator

        parameter.grad = second_gradient.clone()
        optimizer.step()
        torch.testing.assert_close(parameter, after_first - lr * expected_update, rtol=1e-12, atol=1e-12)

        adaptive_mass = memory_weight * (1.0 - beta1) + (1.0 - memory_weight)
        standard_mass = torch.tensor(1.0 - beta1**2, dtype=adaptive_mass.dtype)
        self.assertFalse(torch.isclose(adaptive_mass, standard_mass))

    def test_standard_correction_rejects_norm_matching(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(2))
        with self.assertRaisesRegex(ValueError, 'requires scale="none"'):
            AGAM_AdamW([parameter], scale="step", first_moment_correction="standard")

    def test_old_checkpoint_defaults_to_adaptive_correction(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(2))
        checkpoint = copy.deepcopy(AGAM_AdamW([parameter]).state_dict())
        for group in checkpoint["param_groups"]:
            group.pop("first_moment_correction")

        restored = AGAM_AdamW([torch.nn.Parameter(torch.ones(2))])
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.param_groups[0]["first_moment_correction"], "adaptive")


if __name__ == "__main__":
    unittest.main()
