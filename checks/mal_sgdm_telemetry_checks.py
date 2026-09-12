"""Numerical checks for observations, tensor provenance, persistence and analysis."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optims.mal_opt import MAL_SGDM
from tasks.mal_sgdm_telemetry import (
    GateRecorder,
    Statistics,
    analyze_run,
    build_model,
    learning_rate_at,
    make_loaders,
    parse_args,
    tensor_metadata,
    write_json,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(2, 2)
        self.fc = nn.Linear(2, 2)
        for i in range(1, 5):
            self.add_module(f"layer{i}", nn.Sequential())


class CIFARFixture(Dataset):
    """Exercise real transforms/splitting without downloading CIFAR data."""

    def __init__(self, root, train, download, transform):
        self.targets = list(range(10)) * 20
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        pixels = (np.arange(32 * 32 * 3).reshape(32, 32, 3) + index).astype(np.uint8)
        return self.transform(Image.fromarray(pixels)), self.targets[index]


class MALTelemetryChecks(unittest.TestCase):
    def test_stepwise_warmup_cosine_matches_benchmark_boundary_semantics(self):
        base, minimum, total, warmup = 0.1, 1e-5, 100, 10
        self.assertAlmostEqual(learning_rate_at(0, total, base, "cosine", warmup, minimum), 0.001)
        self.assertAlmostEqual(learning_rate_at(9, total, base, "cosine", warmup, minimum), 0.0901)
        self.assertAlmostEqual(learning_rate_at(10, total, base, "cosine", warmup, minimum), base)
        expected_last = minimum + (base - minimum) * 0.5 * (1 + np.cos(np.pi * 89 / 90))
        self.assertAlmostEqual(learning_rate_at(99, total, base, "cosine", warmup, minimum), expected_last)
        self.assertEqual(learning_rate_at(57, total, base, "constant", warmup, minimum), base)

    def test_observer_is_passive_and_excluded_from_checkpoints(self):
        for options in ({}, {"weight_decay": 0.05}, {"in_place": True}, {"nesterov": True, "scale": True}):
            with self.subTest(options=options):
                model = TinyModel().double()
                control = copy.deepcopy(model)
                observations = []
                optimizer = MAL_SGDM(model.parameters(), gate_observer=lambda *data, sink=observations: sink.append(data), **options)
                baseline = MAL_SGDM(control.parameters(), **options)
                generator = torch.Generator().manual_seed(91)
                for step in range(5):
                    for i, (p, q) in enumerate(zip(model.parameters(), control.parameters(), strict=True)):
                        gradient = torch.randn(p.shape, dtype=p.dtype, generator=generator)
                        if step == 2:
                            gradient.zero_()
                        p.grad = None if step == 3 and i == 1 else gradient.clone()
                        q.grad = None if p.grad is None else p.grad.clone()
                    optimizer.step()
                    baseline.step()
                self.assertEqual(len(observations), 19)
                for p, q in zip(model.parameters(), control.parameters(), strict=True):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                    torch.testing.assert_close(optimizer.state[p]["momentum_buffer"], baseline.state[q]["momentum_buffer"], rtol=0, atol=0)
                checkpoint = optimizer.state_dict()
                self.assertNotIn("gate_observer", optimizer.defaults)
                self.assertTrue(all("gate_observer" not in group for group in checkpoint["param_groups"]))
                observer = optimizer.gate_observer
                optimizer.load_state_dict(checkpoint)
                self.assertIs(optimizer.gate_observer, observer)

    def test_known_alignment_cases_and_applied_update(self):
        parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        observations = []
        optimizer = MAL_SGDM([parameter], gate_observer=lambda _, *values: observations.append(torch.stack(values)))
        # First-step self-alignment, then conflict, then no alignment evidence.
        for gradient, expected_cosine, expected_q, expected_c in (
            (1.0, 1.0, 1.0, 0.9),
            (-0.1, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.9),
        ):
            previous = parameter.detach().clone()
            memory = optimizer.state[parameter].get("momentum_buffer", torch.zeros_like(parameter)).clone()
            parameter.grad = torch.full_like(parameter, gradient)
            optimizer.step()
            actual = observations[-1]
            torch.testing.assert_close(actual[:3], torch.tensor([expected_cosine, expected_q, expected_c], dtype=torch.float64))
            torch.testing.assert_close(parameter, previous - 0.1 * (gradient + actual[2] * memory), rtol=1e-12, atol=1e-12)
        parameter.grad = None
        optimizer.step()
        self.assertEqual(len(observations), 3)

    def test_metadata_keeps_module_kind_role_depth_and_optimizer_group(self):
        model = build_model("group")
        optimizer = MAL_SGDM(model.parameters(), weight_decay=0.01)
        rows = tensor_metadata(model, optimizer)
        by_name = {row["name"]: row for row in rows}
        self.assertEqual(len(rows), 62)
        self.assertEqual([row["tensor_id"] for row in rows], list(range(62)))
        self.assertEqual(sum(row["kind"] == "conv_weight" for row in rows), 20)
        self.assertEqual(sum(row["kind"] == "norm_bias" for row in rows), 20)
        self.assertEqual(by_name["bn1.weight"]["module_type"], "GroupNorm")
        self.assertEqual(by_name["conv1.weight"]["depth_index"], 0)
        self.assertEqual(by_name["fc.bias"]["depth_index"], 9)
        shortcut = by_name["layer2.0.downsample.0.weight"]
        self.assertEqual((shortcut["depth"], shortcut["depth_index"], shortcut["branch"]), ("layer2.0", 3, "shortcut"))
        self.assertEqual(by_name["fc.bias"]["weight_decay"], 0)
        self.assertEqual(shortcut["weight_decay"], 0.01)
        self.assertNotEqual(shortcut["optimizer_group"], by_name["fc.bias"]["optimizer_group"])

    def test_thresholds_include_both_boundaries_in_middle_bin(self):
        values = np.ones((6, 1, 5), dtype=np.float64)
        values[:, 0, 1:3] = np.array([0.0, 0.49, 0.5, 0.7, 0.70001, 1.0])[:, None]
        stats = Statistics(1)
        stats.add(values, np.ones((6, 1), dtype=bool))
        rows, hist = stats.rows({"model": {"all": [0]}}, period="full")
        for row in rows:
            self.assertEqual(row["count"], 6)
            for key in ("pct_lt_0_5", "pct_0_5_to_0_7", "pct_gt_0_7"):
                self.assertAlmostEqual(row[key], 100 / 3)
        self.assertEqual(sum(row["count"] for row in hist if row["metric"] == "gate_q"), 6)

    def test_equal_tensor_and_parameter_count_weighting_are_distinct(self):
        values = np.ones((1, 2, 5), dtype=np.float64)
        values[0, :, 1] = [0.2, 0.8]
        values[0, :, 2] = [0.18, 0.72]
        stats = Statistics(2)
        stats.add(values, np.ones((1, 2), dtype=bool))
        groups = {"model": {"all": [0, 1]}}
        equal, _ = stats.rows(groups, weighting="equal_tensor")
        weighted, _ = stats.rows(groups, tensor_weights=np.array([1, 3]), weighting="numel_weighted")
        equal_q = next(row for row in equal if row["metric"] == "gate_q")
        weighted_q = next(row for row in weighted if row["metric"] == "gate_q")
        self.assertAlmostEqual(equal_q["mean"], 0.5)
        self.assertAlmostEqual(weighted_q["mean"], 0.65)
        self.assertEqual((equal_q["count"], equal_q["weighted_count"]), (2, 2))
        self.assertEqual((weighted_q["count"], weighted_q["weighted_count"]), (2, 4))

    def test_missing_zero_and_norm_floor_are_distinguished(self):
        values = np.ones((5, 1, 5), dtype=np.float64)
        observed = np.ones((5, 1), dtype=bool)
        observed[0] = False
        values[0] = np.nan
        values[1, 0] = [0.0, 0.5, 0.9, 0.0, 1.0]  # Zero gradient: actual c falls back to beta.
        values[2, 0] = [0.0, 0.5, 0.45, 1.0, 0.0]  # Zero probe: no gradient fallback.
        values[3, 0] = [0.001, 0.5005, 0.45045, 1e-12, 1.0]  # Positive norm below implementation floor.
        stats = Statistics(1)
        stats.add(values, observed)
        rows, _ = stats.rows({"model": {"all": [0]}})
        q, coefficient = rows
        self.assertEqual((q["count"], coefficient["count"]), (2, 4))
        self.assertEqual(q["missing_grad"], 1)
        self.assertEqual(q["undefined_alignment"], 2)
        self.assertEqual(q["zero_gradient_fallback"], 1)
        self.assertEqual(q["epsilon_clamped_alignment"], 1)

    def test_cifar_preprocessing_and_saved_split(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tasks.mal_sgdm_telemetry.datasets.CIFAR10", CIFARFixture):
            directory = Path(temporary)
            args = parse_args(["train", "--batch-size", "4"])
            train, validation, test = make_loaders(args, directory, torch.device("cpu"))
            self.assertEqual((len(train.dataset), len(validation.dataset), len(test.dataset)), (170, 30, 200))
            with np.load(directory / "split_indices.npz") as split:
                self.assertEqual(len(np.intersect1d(split["train"], split["validation"])), 0)
                np.testing.assert_array_equal(np.sort(np.concatenate([split["train"], split["validation"]])), np.arange(200))
            for loader in (train, validation, test):
                images, labels = next(iter(loader))
                self.assertEqual(images.shape, (4, 3, 32, 32))
                self.assertEqual(images.dtype, torch.float32)
                self.assertTrue(torch.isfinite(images).all())
                self.assertTrue(((labels >= 0) & (labels < 10)).all())

    def test_failed_optimizer_step_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = TinyModel()
            optimizer = MAL_SGDM(model.parameters())
            metadata = tensor_metadata(model, optimizer)
            recorder = GateRecorder(directory, model, metadata, flush_steps=10)
            optimizer.gate_observer = recorder
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            recorder.begin_step()
            optimizer.step()
            recorder.end_step(epoch=1, batch=1, batch_size=2, samples_seen=2, learning_rates=[0.1], loss=torch.tensor(1.0))

            def interrupted_observer(parameter, *values):
                recorder(parameter, *values)
                if parameter is model.conv1.bias:
                    raise RuntimeError("Simulated failure partway through optimizer.step")

            optimizer.gate_observer = interrupted_observer
            recorder.begin_step()
            with self.assertRaisesRegex(RuntimeError, "Simulated failure"):
                optimizer.step()
            recorder.abort_step()
            recorder.flush()
            self.assertEqual(recorder.completed_steps, 1)
            index = json.loads((directory / "index.json").read_text())
            self.assertEqual(index["completed_steps"], 1)
            with np.load(directory / index["chunks"][0]["file"]) as data:
                np.testing.assert_array_equal(data["tensor_step"], np.ones((1, 4), dtype=int))
                self.assertEqual(data["values"].shape, (1, 4, 5))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                GateRecorder(directory, model, metadata)

    def test_shards_preserve_exact_applied_coefficients_and_late_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = TinyModel().double()
            optimizer = MAL_SGDM(model.parameters(), weight_decay=0.01)
            metadata = tensor_metadata(model, optimizer)
            recorder = GateRecorder(directory / "telemetry", model, metadata, flush_steps=2)
            optimizer.gate_observer = recorder
            write_json(directory / "tensor_metadata.json", metadata)
            write_json(directory / "run.json", {"status": "completed", "synthetic": True, "norm": "group"})
            parameters = list(model.parameters())
            before, memories, gradients = [], [], []
            for step in range(1, 6):
                for i, parameter in enumerate(parameters):
                    parameter.grad = None if step == 2 and i == 3 else torch.full_like(parameter, 0.2)
                parameters[1].grad = torch.zeros_like(parameters[1]) if step == 3 else parameters[1].grad
                before.append([p.detach().clone() for p in parameters])
                memories.append([optimizer.state[p].get("momentum_buffer", torch.zeros_like(p)).clone() for p in parameters])
                gradients.append([None if p.grad is None else p.grad.add(p, alpha=row["weight_decay"]) for p, row in zip(parameters, metadata, strict=True)])
                recorder.begin_step()
                optimizer.step()
                recorder.end_step(
                    epoch=1 if step <= 3 else 2,
                    batch=step,
                    batch_size=2,
                    samples_seen=step * 2,
                    learning_rates=[g["lr"] for g in optimizer.param_groups],
                    loss=torch.tensor(1.0, dtype=torch.float64),
                )
                before[-1] = [old - p.detach() for old, p in zip(before[-1], parameters, strict=True)]
            recorder.flush()
            index = json.loads((directory / "telemetry" / "index.json").read_text())
            self.assertEqual([entry["steps"] for entry in index["chunks"]], [2, 2, 1])
            chunks = []
            for entry in index["chunks"]:
                with np.load(directory / "telemetry" / entry["file"]) as data:
                    chunks.append(data["values"].copy())
            values = np.concatenate(chunks)
            self.assertTrue(np.isnan(values[1, 3]).all())
            np.testing.assert_array_equal(values[2, 1, 1:3], [0.5, 0.9])
            for step in range(5):
                for i in range(len(parameters)):
                    gradient = gradients[step][i]
                    if gradient is not None:
                        expected_delta = 0.1 * (gradient + float(values[step, i, 2]) * memories[step][i])
                        torch.testing.assert_close(before[step][i], expected_delta, rtol=1e-12, atol=1e-12)
            analyze_run(directory, late_fraction=0.4, plots=False)
            analysis = json.loads((directory / "analysis" / "analysis.json").read_text())
            self.assertEqual((analysis["late_first_step"], analysis["late_last_step"], analysis["late_steps"]), (4, 5, 2))
            import csv

            with (directory / "analysis" / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            full = next(row for row in rows if row["metric"] == "beta_eff" and row["period"] == "full" and row["view"] == "model")
            self.assertEqual(int(full["count"]), 19)
            late = next(row for row in rows if row["metric"] == "gate_q" and row["period"] == "late" and row["view"] == "model")
            self.assertEqual(int(late["count"]), 8)
            self.assertAlmostEqual(float(late["mean"]), values[3:, :, 1].mean())


if __name__ == "__main__":
    unittest.main()
