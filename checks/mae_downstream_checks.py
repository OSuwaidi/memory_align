"""Focused structural checks for MAE downstream evaluation and its sweep."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sweeps.agam_lion_mae_extended_sweep import EXPECTED_RUNS, build_sweep
from sweeps.mae_lion_finetune_confirmation_sweep import (
    EXPECTED_CONFIRMATION_RUNS,
    select_agam_sources,
)
from sweeps.mae_lion_finetune_confirmation_sweep import (
    build_sweep as build_finetune_sweep,
)
from tasks.mae_pretrain import MAEEncoderClassifier, MaskedAutoencoderViT, layerwise_finetune_param_groups


class MAEDownstreamChecks(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.mae = MaskedAutoencoderViT(
            image_size=32,
            patch_size=8,
            decoder_embed_dim=32,
            decoder_depth=1,
            decoder_num_heads=4,
        )

    def test_classifier_reuses_only_encoder_and_backpropagates(self) -> None:
        classifier = MAEEncoderClassifier(self.mae, num_classes=10, drop_path_rate=0.1)
        logits = classifier(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 10))
        logits.square().mean().backward()

        classifier_ids = {id(parameter) for parameter in classifier.parameters()}
        decoder_ids = {
            id(parameter)
            for module in (
                self.mae.decoder_embed,
                self.mae.decoder_blocks,
                self.mae.decoder_norm,
                self.mae.decoder_pred,
            )
            for parameter in module.parameters()
        }
        self.assertTrue(classifier_ids.isdisjoint(decoder_ids))
        self.assertTrue(self.mae.pos_embed.requires_grad)
        self.assertIsNotNone(self.mae.pos_embed.grad)

    def test_layerwise_groups_cover_each_trainable_parameter_once(self) -> None:
        classifier = MAEEncoderClassifier(self.mae, num_classes=10, drop_path_rate=0.1)
        groups = layerwise_finetune_param_groups(classifier, weight_decay=0.05, layer_decay=0.65)
        grouped = [parameter for group in groups for parameter in group["params"]]
        expected = [parameter for parameter in classifier.parameters() if parameter.requires_grad]

        self.assertEqual(len(grouped), len(expected))
        self.assertEqual({id(parameter) for parameter in grouped}, {id(parameter) for parameter in expected})
        self.assertEqual(len({id(parameter) for parameter in grouped}), len(grouped))
        self.assertEqual({group["weight_decay"] for group in groups}, {0.0, 0.05})
        self.assertLess(min(group["lr_scale"] for group in groups), 1.0)
        self.assertEqual(max(group["lr_scale"] for group in groups), 1.0)

    def test_extended_sweep_has_exact_requested_grid(self) -> None:
        args = argparse.Namespace(
            program="tasks/mae_pretrain.py",
            sweep_name="structural-check",
            source_revision="test-revision",
            data_dir="/tmp/tiny-imagenet-200",
            output_dir="/tmp/agam-lion-mae",
            epochs=300,
            warmup_epochs=15,
            probe_every=50,
        )
        sweep = build_sweep(args)
        cardinality = 1
        for parameter in sweep["parameters"].values():
            cardinality *= len(parameter["values"])

        self.assertEqual(cardinality, EXPECTED_RUNS)
        self.assertEqual(EXPECTED_RUNS, 18)
        self.assertEqual(sweep["parameters"]["optimizer"]["values"], ("AGAM_Lion",))
        self.assertEqual(sweep["parameters"]["base_lr"]["values"], (2e-4, 1.5e-4, 5e-5))
        self.assertEqual(sweep["parameters"]["weight_decay"]["values"], (0.5, 0.25, 0.15))
        self.assertEqual(sweep["parameters"]["seed"]["values"], (42, 1337))
        self.assertEqual(sweep["metric"]["name"], "linear_probe/final_val_top1_pct")
        self.assertNotIn("--run_finetune", sweep["command"])

    def test_finetune_sweep_has_one_run_per_selected_checkpoint(self) -> None:
        source_ids = ["lion42", "lion1337", "agam42", "agam1337"]
        args = argparse.Namespace(
            program="tasks/mae_finetune_eval.py",
            sweep_name="structural-check",
            source_revision="test-revision",
            data_dir="/tmp/tiny-imagenet-200",
            project_name="MAL_benchmark",
        )
        sweep = build_finetune_sweep(args, source_ids)
        cardinality = 1
        for parameter in sweep["parameters"].values():
            cardinality *= len(parameter["values"])

        self.assertEqual(EXPECTED_CONFIRMATION_RUNS, 4)
        self.assertEqual(cardinality, EXPECTED_CONFIRMATION_RUNS)
        self.assertEqual(sweep["parameters"]["source_run_id"]["values"], source_ids)
        self.assertEqual(sweep["metric"]["name"], "finetune/final_val_top1_pct")

    def test_final_screen_grid_is_exactly_six_runs(self) -> None:
        args = argparse.Namespace(
            program="tasks/mae_pretrain.py",
            sweep_name="final-screen-check",
            source_revision="test-revision",
            data_dir="/tmp/tiny-imagenet-200",
            output_dir="/tmp/agam-lion-mae-final-screen",
            epochs=300,
            warmup_epochs=15,
            probe_every=50,
            base_lrs=(1e-4,),
            weight_decays=(0.3, 0.5, 0.75),
            seeds=(42, 1337),
            comparison_group="agam_lion_mae_final_probe_screen_v1",
            study_stage="agam_lion_mae_final_probe_screen",
        )
        sweep = build_sweep(args)
        cardinality = 1
        for parameter in sweep["parameters"].values():
            cardinality *= len(parameter["values"])

        self.assertEqual(cardinality, 6)
        self.assertEqual(sweep["parameters"]["base_lr"]["values"], (1e-4,))
        self.assertEqual(sweep["parameters"]["weight_decay"]["values"], (0.3, 0.5, 0.75))
        self.assertEqual(sweep["parameters"]["seed"]["values"], (42, 1337))

    def test_combined_screens_select_paired_seed_mean(self) -> None:
        cells = (
            (2e-4, 0.5, (20.0, 21.0)),
            (2e-4, 0.25, (22.0, 23.0)),
            (2e-4, 0.15, (24.0, 25.0)),
            (1.5e-4, 0.5, (26.0, 27.0)),
            (1.5e-4, 0.25, (28.0, 29.0)),
            (1.5e-4, 0.15, (27.0, 28.0)),
            (5e-5, 0.5, (25.0, 26.0)),
            (5e-5, 0.25, (24.0, 25.0)),
            (5e-5, 0.15, (23.0, 24.0)),
            (1e-4, 0.3, (29.0, 30.0)),
            (1e-4, 0.5, (31.0, 32.0)),
            (1e-4, 0.75, (30.0, 31.0)),
        )
        runs = []
        with TemporaryDirectory() as temporary_directory:
            for cell_index, (base_lr, weight_decay, accuracies) in enumerate(cells):
                for seed, accuracy in zip((42, 1337), accuracies, strict=True):
                    checkpoint = Path(temporary_directory) / f"{cell_index}-{seed}.pt"
                    checkpoint.touch()
                    runs.append(
                        SimpleNamespace(
                            id=f"run-{cell_index}-{seed}",
                            state="finished",
                            config={
                                "optimizer": "AGAM_Lion",
                                "base_lr": base_lr,
                                "weight_decay": weight_decay,
                                "seed": seed,
                                "epochs": 300,
                            },
                            summary={
                                "linear_probe/final_val_top1_pct": accuracy,
                                "checkpoint": str(checkpoint),
                                "epoch": 300,
                            },
                        )
                    )

            selected, ranking = select_agam_sources(runs, expected_runs=24)

        self.assertEqual((ranking[0]["base_lr"], ranking[0]["weight_decay"]), (1e-4, 0.5))
        self.assertEqual([run.config["seed"] for run in selected], [42, 1337])

    def test_selection_ignores_interrupted_runs_when_recovered(self) -> None:
        runs = []
        with TemporaryDirectory() as temporary_directory:
            for cell_index in range(12):
                base_lr = 1e-4 + cell_index * 1e-6
                for seed in (42, 1337):
                    checkpoint = Path(temporary_directory) / f"{cell_index}-{seed}.pt"
                    checkpoint.touch()
                    runs.append(
                        SimpleNamespace(
                            id=f"finished-{cell_index}-{seed}",
                            state="finished",
                            config={
                                "optimizer": "AGAM_Lion",
                                "base_lr": base_lr,
                                "weight_decay": 0.5,
                                "seed": seed,
                                "epochs": 300,
                            },
                            summary={
                                "linear_probe/final_val_top1_pct": float(cell_index),
                                "checkpoint": str(checkpoint),
                                "epoch": 300,
                            },
                        )
                    )
            runs.append(
                SimpleNamespace(
                    id="interrupted-source",
                    state="crashed",
                    config={
                        "optimizer": "AGAM_Lion",
                        "base_lr": 5e-5,
                        "weight_decay": 0.15,
                        "seed": 42,
                    },
                    summary={},
                )
            )

            selected, ranking = select_agam_sources(runs, expected_runs=24)

        self.assertEqual(len(ranking), 12)
        self.assertEqual([run.config["seed"] for run in selected], [42, 1337])

    def test_selection_excludes_an_incomplete_cell(self) -> None:
        runs = []
        with TemporaryDirectory() as temporary_directory:
            for cell_index, seeds in enumerate(((42, 1337), (42,))):
                for seed in seeds:
                    checkpoint = Path(temporary_directory) / f"{cell_index}-{seed}.pt"
                    checkpoint.touch()
                    runs.append(
                        SimpleNamespace(
                            id=f"run-{cell_index}-{seed}",
                            state="finished",
                            config={
                                "optimizer": "AGAM_Lion",
                                "base_lr": 1e-4 + cell_index * 1e-4,
                                "weight_decay": 0.5,
                                "seed": seed,
                                "epochs": 300,
                            },
                            summary={
                                "final/probe_val_acc": 25.0 + cell_index,
                                "checkpoint": str(checkpoint),
                                "epoch": 300,
                            },
                        )
                    )

            selected, ranking = select_agam_sources(runs, expected_runs=3)

        self.assertEqual(len(ranking), 1)
        self.assertEqual([run.config["seed"] for run in selected], [42, 1337])


if __name__ == "__main__":
    unittest.main()
