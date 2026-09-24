"""Numerical and schema checks for AGAM-AdamW MAE gate telemetry."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analysis.analyze_agam_adamw_mae_telemetry import analyze
from optims.agam_opt import AGAM_AdamW
from tasks.agam_adamw_mae_telemetry import ViTGateRecorder, telemetry_manifest, vit_tensor_metadata
from tasks.mae_pretrain import train_one_epoch


class MiniAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)


class MiniMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)


class MiniBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = MiniAttention(width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = MiniMLP(width)


class MiniPatchEmbed(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, width, kernel_size=2, stride=2)


class MiniMAE(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.patch_embed = MiniPatchEmbed(width)
        self.blocks = nn.ModuleList([MiniBlock(width), MiniBlock(width)])
        self.norm = nn.LayerNorm(width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.decoder_embed = nn.Linear(width, width)
        self.decoder_blocks = nn.ModuleList([MiniBlock(width)])
        self.decoder_norm = nn.LayerNorm(width)
        self.decoder_pred = nn.Linear(width, 12)

    def forward(self, images: torch.Tensor, _mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss = images.square().mean() * 0
        for parameter in self.parameters():
            loss = loss + parameter.square().mean()
        empty = images.new_empty(0)
        return loss, empty, empty


class AGAMAdamWTelemetryChecks(unittest.TestCase):
    def test_observer_is_passive_and_reports_exact_gate(self) -> None:
        parameter = nn.Parameter(torch.tensor([[0.3, -0.5], [0.7, 0.2]], dtype=torch.float64))
        reference = nn.Parameter(parameter.detach().clone())
        observations: list[tuple[torch.Tensor, ...]] = []
        observed = AGAM_AdamW(
            [parameter],
            lr=0.03,
            betas=(0.9, 0.95),
            weight_decay=0.02,
            gate_observer=lambda _parameter, *values: observations.append(tuple(value.clone() for value in values)),
        )
        control = AGAM_AdamW([reference], lr=0.03, betas=(0.9, 0.95), weight_decay=0.02)
        for gradient in (
            torch.tensor([[0.2, -0.1], [0.4, -0.3]], dtype=torch.float64),
            torch.tensor([[-0.5, 0.2], [-0.1, 0.6]], dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
        ):
            parameter.grad = gradient.clone()
            reference.grad = gradient.clone()
            observed.step()
            control.step()
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq", "first_moment_weight"):
            torch.testing.assert_close(observed.state[parameter][key], control.state[reference][key], rtol=0, atol=0)
        self.assertEqual(len(observations), 3)
        for cosine, gate, beta_eff, _gradient_norm, _probe_norm in observations:
            torch.testing.assert_close(gate, (1 + cosine) / 2, rtol=0, atol=0)
            if torch.isfinite(gate) and _gradient_norm > 0:
                torch.testing.assert_close(beta_eff, 0.9 * gate, rtol=0, atol=0)
        self.assertNotIn("gate_observer", observed.defaults)
        self.assertNotIn("gate_observer", observed.state_dict()["param_groups"][0])

    def test_vit_taxonomy_is_complete_and_analysis_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = MiniMAE().double()
            optimizer = AGAM_AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.05)
            metadata = vit_tensor_metadata(model, optimizer)
            self.assertEqual(len(metadata), sum(1 for _ in model.parameters()))
            self.assertEqual(len({row["name"] for row in metadata}), len(metadata))
            self.assertIn("attention_qkv_weight", {row["kind"] for row in metadata})
            self.assertIn("normalization_scale", {row["kind"] for row in metadata})
            self.assertIn("encoder.block.00", {row["depth"] for row in metadata})
            self.assertIn("decoder.reconstruction_head", {row["depth"] for row in metadata})
            (root / "tensor_metadata.json").write_text(json.dumps(metadata))
            (root / "telemetry_manifest.json").write_text(json.dumps(telemetry_manifest(metadata)))
            recorder = ViTGateRecorder(root / "telemetry", model, metadata, flush_steps=2)
            optimizer.gate_observer = recorder
            generator = torch.Generator().manual_seed(2027)
            for step in range(4):
                for parameter in model.parameters():
                    parameter.grad = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                    if step == 2:
                        parameter.grad.neg_()
                recorder.begin_step()
                optimizer.step()
                recorder.end_step(
                    epoch=1 + step // 2,
                    batch=1 + step % 2,
                    batch_size=8,
                    samples_seen=8 * (step + 1),
                    learning_rates=[group["lr"] for group in optimizer.param_groups],
                    loss=torch.tensor(1.0 / (step + 1), dtype=torch.float64),
                )
            recorder.flush()
            output = analyze(root, plots=True)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["completed_steps"], 4)
            self.assertEqual(summary["tensor_count"], len(metadata))
            self.assertTrue((output / "tables" / "epoch_group_gate_statistics.csv").exists())
            self.assertTrue((output / "tables" / "step_global_gate_counterfactual.csv").exists())
            self.assertTrue((output / "figures" / "gate_evolution_overview.pdf").exists())
            self.assertTrue((output / "figures" / "gate_depth_heatmaps.png").exists())

    def test_mae_epoch_training_commits_only_completed_optimizer_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = MiniMAE().float()
            optimizer = AGAM_AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
            metadata = vit_tensor_metadata(model, optimizer)
            recorder = ViTGateRecorder(Path(temporary) / "telemetry", model, metadata, flush_steps=8)
            optimizer.gate_observer = recorder
            loader = DataLoader(
                TensorDataset(torch.randn(4, 3, 4, 4), torch.zeros(4, dtype=torch.long)),
                batch_size=2,
            )
            loss, final_lr = train_one_epoch(
                model,  # type: ignore[arg-type]
                optimizer,
                loader,
                device=torch.device("cpu"),
                epoch=1,
                epochs=2,
                steps_per_epoch=2,
                accumulation_steps=1,
                mask_ratio=0.75,
                peak_lr=1e-3,
                min_lr=0.0,
                warmup_epochs=0,
                use_scheduler=False,
                amp_dtype=torch.float32,
                amp_enabled=False,
                telemetry_recorder=recorder,
            )
            recorder.flush()
            aggregates = recorder.pop_epoch_aggregates(1)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertEqual(final_lr, 1e-3)
            self.assertEqual(recorder.completed_steps, 2)
            self.assertIn("telemetry/model/all/equal_tensor/gate_mean", aggregates)
            self.assertIn("telemetry/depth/encoder.block.00/equal_tensor/misalignment_pct", aggregates)


if __name__ == "__main__":
    unittest.main()
