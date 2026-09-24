"""Numerical and configuration checks for the AGAM-AdamW component ablation."""

from __future__ import annotations

import argparse
import copy
import math
import unittest

import torch

from optims.agam_opt import AGAM_AdamW
from sweeps.agam_adamw_component_ablation_sweep import EXPECTED_RUNS, build_sweep


def optimizer(parameters, **overrides) -> AGAM_AdamW:
    settings = {
        "lr": 0.01,
        "betas": (0.9, 0.95),
        "eps": 1e-7,
        "weight_decay": 0.0,
        "pwr": 1.0,
        "align": "update",
        "in_place": False,
        "scale": "none",
        "gate_mode": "attenuate",
        "gradient_weight_mode": "complement",
        "first_moment_correction": "adaptive",
    }
    settings.update(overrides)
    return AGAM_AdamW(parameters, **settings)


class AGAMAdamWComponentChecks(unittest.TestCase):
    def test_sweep_has_exact_matched_cardinality_and_recipe(self) -> None:
        args = argparse.Namespace(
            program="tasks/mae_pretrain.py",
            sweep_name="test",
            data_dir="/data/tiny-imagenet-200",
            output_dir="/outputs/mae",
            source_revision="deadbeef",
        )
        sweep = build_sweep(args)
        cardinality = math.prod(len(parameter["values"]) for parameter in sweep["parameters"].values())
        self.assertEqual(cardinality, EXPECTED_RUNS)
        self.assertEqual(EXPECTED_RUNS, 15)
        self.assertEqual(sweep["parameters"]["batch_size"]["values"], (1024,))
        self.assertEqual(sweep["parameters"]["base_lr"]["values"], (1e-3,))
        self.assertEqual(sweep["parameters"]["weight_decay"]["values"], (5e-2,))
        self.assertEqual(sweep["parameters"]["seed"]["values"], (42, 1337, 2026))
        self.assertEqual(
            set(sweep["parameters"]["AGAM_variant"]["values"]),
            {"canonical", "previous_memory", "global_gate", "writeback", "hard_reset"},
        )
        command = sweep["command"]
        self.assertEqual(command[command.index("--epochs") + 1], "300")
        self.assertEqual(command[command.index("--warmup_epochs") + 1], "15")
        self.assertEqual(command[command.index("--beta2") + 1], "0.95")

    def test_explicit_canonical_controls_preserve_default_update(self) -> None:
        left = torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float64))
        right = torch.nn.Parameter(left.detach().clone())
        default = optimizer([left])
        explicit = optimizer([right], alignment_source="probe", gate_scope="tensor", conflict_strategy="soft")
        for gradient in (
            torch.tensor([0.4, -0.1], dtype=torch.float64),
            torch.tensor([-0.2, 0.3], dtype=torch.float64),
            torch.tensor([0.1, 0.15], dtype=torch.float64),
        ):
            left.grad = gradient.clone()
            right.grad = gradient.clone()
            default.step()
            explicit.step()
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        left_state = default.state[left]
        right_state = explicit.state[right]
        for key in ("exp_avg", "exp_avg_sq", "first_moment_weight"):
            torch.testing.assert_close(left_state[key], right_state[key], rtol=0.0, atol=0.0)

    def test_previous_memory_changes_only_alignment_source(self) -> None:
        probe_parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
        memory_parameter = torch.nn.Parameter(probe_parameter.detach().clone())
        observed: dict[str, float] = {}
        probe = optimizer(
            [probe_parameter],
            gate_observer=lambda _p, _c, gate, *_rest: observed.__setitem__("probe", float(gate)),
        )
        previous = optimizer(
            [memory_parameter],
            alignment_source="memory",
            gate_observer=lambda _p, _c, gate, *_rest: observed.__setitem__("memory", float(gate)),
        )
        gradient = torch.tensor([1.0, -2.0], dtype=torch.float64)
        probe_parameter.grad = gradient.clone()
        memory_parameter.grad = gradient.clone()
        probe.step()
        previous.step()
        # AdamW update-space alignment compares the raw gradient with its
        # preconditioned probe, so even the first-step gate need not equal one.
        self.assertGreater(observed["probe"], observed["memory"])
        self.assertAlmostEqual(observed["memory"], 0.5)
        for optim, parameter in ((probe, probe_parameter), (previous, memory_parameter)):
            torch.testing.assert_close(optim.state[parameter]["exp_avg"], 0.1 * gradient, rtol=1e-14, atol=1e-14)

    def test_hard_reset_is_transient_and_preserves_base_memory(self) -> None:
        soft_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        reset_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        soft = optimizer([soft_parameter])
        reset = optimizer([reset_parameter], conflict_strategy="hard_reset")
        for optim, parameter in ((soft, soft_parameter), (reset, reset_parameter)):
            optim.state[parameter] = {
                "step": 1,
                "exp_avg": torch.tensor([1.0, 0.0], dtype=torch.float64),
                "exp_avg_sq": torch.tensor([0.2, 0.2], dtype=torch.float64),
                "first_moment_weight": torch.tensor(0.1, dtype=torch.float64),
            }
            parameter.grad = torch.tensor([-0.2, 0.2], dtype=torch.float64)
        soft.step()
        reset.step()
        expected_probe = torch.tensor([0.88, 0.02], dtype=torch.float64)
        torch.testing.assert_close(soft.state[soft_parameter]["exp_avg"], expected_probe)
        torch.testing.assert_close(reset.state[reset_parameter]["exp_avg"], expected_probe)
        self.assertFalse(torch.equal(soft_parameter, reset_parameter))

    def test_writeback_changes_persistent_first_moment_only_as_intended(self) -> None:
        transient_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        writeback_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        transient = optimizer([transient_parameter])
        writeback = optimizer([writeback_parameter], in_place=True)
        first_gradient = torch.tensor([1.0, 0.0], dtype=torch.float64)
        second_gradient = torch.tensor([-0.2, 0.2], dtype=torch.float64)
        for gradient in (first_gradient, second_gradient):
            transient_parameter.grad = gradient.clone()
            writeback_parameter.grad = gradient.clone()
            transient.step()
            writeback.step()
        self.assertFalse(
            torch.equal(
                transient.state[transient_parameter]["exp_avg"],
                writeback.state[writeback_parameter]["exp_avg"],
            )
        )

    def test_global_gate_is_shared_and_is_not_mean_tensor_gate(self) -> None:
        parameter_a = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        parameter_b = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        observed: list[float] = []
        global_optimizer = optimizer(
            [parameter_a, parameter_b],
            gate_scope="global",
            gate_observer=lambda _p, _c, gate, *_rest: observed.append(float(gate)),
        )
        for parameter, memory, second_moment in (
            (parameter_a, torch.tensor([1.0, 0.0]), torch.tensor([0.2, 0.2])),
            (parameter_b, torch.tensor([0.0, 2.0]), torch.tensor([0.5, 0.1])),
        ):
            global_optimizer.state[parameter] = {
                "step": 1,
                "exp_avg": memory.to(torch.float64),
                "exp_avg_sq": second_moment.to(torch.float64),
                "first_moment_weight": torch.tensor(0.1, dtype=torch.float64),
            }
        parameter_a.grad = torch.tensor([-0.2, 0.2], dtype=torch.float64)
        parameter_b.grad = torch.tensor([0.3, -0.1], dtype=torch.float64)
        global_optimizer.step()
        self.assertEqual(len(observed), 2)
        self.assertAlmostEqual(observed[0], observed[1], places=14)

    def test_global_gate_matches_one_flattened_tensor(self) -> None:
        left = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        right = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        flat = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
        global_optimizer = optimizer([left, right], gate_scope="global")
        flat_optimizer = optimizer([flat])
        memories = (
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([0.0, 2.0, -0.5], dtype=torch.float64),
        )
        second_moments = (
            torch.tensor([0.2], dtype=torch.float64),
            torch.tensor([0.5, 0.1, 0.3], dtype=torch.float64),
        )
        gradients = (
            torch.tensor([-0.2], dtype=torch.float64),
            torch.tensor([0.3, -0.1, 0.4], dtype=torch.float64),
        )
        for parameter, memory, second_moment, gradient in zip((left, right), memories, second_moments, gradients, strict=True):
            global_optimizer.state[parameter] = {
                "step": 1,
                "exp_avg": memory.clone(),
                "exp_avg_sq": second_moment.clone(),
                "first_moment_weight": torch.tensor(0.1, dtype=torch.float64),
            }
            parameter.grad = gradient.clone()
        flat_optimizer.state[flat] = {
            "step": 1,
            "exp_avg": torch.cat(memories),
            "exp_avg_sq": torch.cat(second_moments),
            "first_moment_weight": torch.tensor(0.1, dtype=torch.float64),
        }
        flat.grad = torch.cat(gradients)
        global_optimizer.step()
        flat_optimizer.step()
        torch.testing.assert_close(torch.cat((left, right)), flat, rtol=1e-13, atol=1e-13)

    def test_old_checkpoint_loads_canonical_component_defaults(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        source = optimizer([parameter])
        state = copy.deepcopy(source.state_dict())
        for group in state["param_groups"]:
            group.pop("alignment_source")
            group.pop("gate_scope")
            group.pop("conflict_strategy")
        target_parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        target = optimizer([target_parameter], alignment_source="memory", gate_scope="global")
        target.load_state_dict(state)
        for group in target.param_groups:
            self.assertEqual(group["alignment_source"], "probe")
            self.assertEqual(group["gate_scope"], "tensor")
            self.assertEqual(group["conflict_strategy"], "soft")


if __name__ == "__main__":
    unittest.main()
