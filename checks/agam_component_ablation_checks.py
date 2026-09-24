"""Numerical checks for the paper's AGAM-SGD component ablations."""

from __future__ import annotations

import copy
import unittest

import torch

from optims.agam_opt import AGAM_SGD


DTYPE = torch.float64


def install_memory(optimizer: AGAM_SGD, parameter: torch.nn.Parameter, memory: torch.Tensor) -> None:
    optimizer.state[parameter]["momentum_buffer"] = memory.clone()
    optimizer.state[parameter]["step"] = 0


class AGAMComponentAblationChecks(unittest.TestCase):
    def test_canonical_update_is_unchanged(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=DTYPE))
        gradient = torch.tensor([-0.2, 0.5], dtype=DTYPE)
        memory = torch.tensor([0.3, -0.4], dtype=DTYPE)
        optimizer = AGAM_SGD([parameter], lr=0.1, beta=0.9)
        install_memory(optimizer, parameter, memory)
        parameter.grad = gradient.clone()

        probe = gradient + 0.9 * memory
        cosine = torch.dot(gradient, probe) / (gradient.norm() * probe.norm())
        gate = (1.0 + cosine) / 2.0
        applied = gradient + 0.9 * gate * memory
        expected_parameter = parameter.detach().clone() - 0.1 * applied

        optimizer.step()
        torch.testing.assert_close(parameter, expected_parameter, rtol=0.0, atol=1e-14)
        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], probe, rtol=0.0, atol=1e-14)

    def test_previous_memory_changes_only_alignment_reference(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=DTYPE))
        gradient = torch.tensor([-0.2, 0.5], dtype=DTYPE)
        memory = torch.tensor([0.3, -0.4], dtype=DTYPE)
        optimizer = AGAM_SGD(
            [parameter],
            lr=0.1,
            beta=0.9,
            alignment_source="memory",
        )
        install_memory(optimizer, parameter, memory)
        parameter.grad = gradient.clone()

        cosine = torch.dot(gradient, memory) / (gradient.norm() * memory.norm())
        gate = (1.0 + cosine) / 2.0
        applied = gradient + 0.9 * gate * memory
        expected_parameter = parameter.detach().clone() - 0.1 * applied
        expected_probe = gradient + 0.9 * memory

        optimizer.step()
        torch.testing.assert_close(parameter, expected_parameter, rtol=0.0, atol=1e-14)
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"], expected_probe, rtol=0.0, atol=1e-14
        )

    def test_writeback_persists_the_gated_moment(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=DTYPE))
        gradient = torch.tensor([-0.2, 0.5], dtype=DTYPE)
        memory = torch.tensor([0.3, -0.4], dtype=DTYPE)
        optimizer = AGAM_SGD([parameter], lr=0.1, beta=0.9, in_place=True)
        install_memory(optimizer, parameter, memory)
        parameter.grad = gradient.clone()

        probe = gradient + 0.9 * memory
        cosine = torch.dot(gradient, probe) / (gradient.norm() * probe.norm())
        applied = gradient + 0.9 * ((1.0 + cosine) / 2.0) * memory
        optimizer.step()
        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], applied, rtol=0.0, atol=1e-14)

    def test_hard_reset_is_transient(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=DTYPE))
        gradient = torch.tensor([1.0, 0.0], dtype=DTYPE)
        desired_probe = torch.tensor([-0.5, 1.0], dtype=DTYPE)
        memory = (desired_probe - gradient) / 0.9
        optimizer = AGAM_SGD(
            [parameter],
            lr=0.1,
            beta=0.9,
            conflict_strategy="hard_reset",
        )
        install_memory(optimizer, parameter, memory)
        parameter.grad = gradient.clone()
        expected_parameter = parameter.detach().clone() - 0.1 * gradient

        optimizer.step()
        torch.testing.assert_close(parameter, expected_parameter, rtol=0.0, atol=1e-14)
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"], desired_probe, rtol=0.0, atol=1e-14
        )

    def test_global_gate_is_cosine_of_concatenated_vectors(self) -> None:
        first = torch.nn.Parameter(torch.tensor([0.0], dtype=DTYPE))
        second = torch.nn.Parameter(torch.tensor([0.0], dtype=DTYPE))
        gradients = (torch.tensor([1.0], dtype=DTYPE), torch.tensor([1.0], dtype=DTYPE))
        memories = (torch.tensor([-2.0], dtype=DTYPE), torch.tensor([2.0], dtype=DTYPE))
        optimizer = AGAM_SGD([first, second], lr=0.1, beta=0.9, gate_scope="global")
        for parameter, gradient, memory in zip((first, second), gradients, memories, strict=True):
            install_memory(optimizer, parameter, memory)
            parameter.grad = gradient.clone()

        probes = tuple(g + 0.9 * m for g, m in zip(gradients, memories, strict=True))
        dot = sum(torch.dot(g, probe) for g, probe in zip(gradients, probes, strict=True))
        gradient_norm = torch.cat(gradients).norm()
        probe_norm = torch.cat(probes).norm()
        gate = (1.0 + dot / (gradient_norm * probe_norm)) / 2.0
        expected = tuple(-0.1 * (g + 0.9 * gate * m) for g, m in zip(gradients, memories, strict=True))

        optimizer.step()
        torch.testing.assert_close(first, expected[0], rtol=0.0, atol=1e-14)
        torch.testing.assert_close(second, expected[1], rtol=0.0, atol=1e-14)
        for parameter, probe in zip((first, second), probes, strict=True):
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"], probe, rtol=0.0, atol=1e-14
            )

    def test_legacy_state_dict_gets_canonical_defaults(self) -> None:
        source_parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=DTYPE))
        source = AGAM_SGD([source_parameter])
        legacy = copy.deepcopy(source.state_dict())
        for group in legacy["param_groups"]:
            group.pop("alignment_source")
            group.pop("gate_scope")
            group.pop("conflict_strategy")

        target_parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=DTYPE))
        target = AGAM_SGD(
            [target_parameter],
            alignment_source="memory",
            gate_scope="global",
            conflict_strategy="hard_reset",
        )
        target.load_state_dict(legacy)
        self.assertEqual(target.param_groups[0]["alignment_source"], "probe")
        self.assertEqual(target.param_groups[0]["gate_scope"], "tensor")
        self.assertEqual(target.param_groups[0]["conflict_strategy"], "soft")


if __name__ == "__main__":
    unittest.main()
