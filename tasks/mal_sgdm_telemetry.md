# MAL-SGDM tensor telemetry

`mal_sgdm_telemetry.py` trains a CIFAR-sized ResNet18 and records the **actual scalar computed by MAL for every trainable tensor on every optimizer step**. It produces numeric source tables and paper-ready PNG/PDF/SVG figures. It runs locally on CPU, Apple MPS or CUDA; W&B logging and artifact upload are optional.

## Run

From the repository root, with its Python environment:

```bash
.venv/bin/python tasks/mal_sgdm_telemetry.py train \
  --output analysis/mal_sgdm_telemetry_cifar10_seed42 \
  --device mps --download
```

This starts a **200-epoch CIFAR-10 run**. `--download` permits torchvision to fetch the dataset; omit it if the dataset is already in `--data-dir` (default: `data/`). On another machine, use `--device cuda` or `--device auto`. Choose an empty output directory for each run.

The optimizer's structural defaults come directly from `MAL_SGDM`: `beta=0.9`, `pwr=1`, `in_place=False`, `scale=False`, `nesterov=False`, `gate_mode="attenuate"`. The default learning rate is `0.1`, **constant**, with **zero weight decay** and no warmup. `--lr`, `--weight-decay`, `--schedule cosine`, and `--warmup-epochs` are explicit training-recipe overrides; each is recorded. Cosine scheduling uses the benchmark's `1e-5` floor (configurable with `--min-lr`); an early `--max-steps` stop does not compress that schedule. Warmup, if enabled, follows the benchmark's step-wise linear schedule from 1% of the base rate.

The canonical diagnostics run matching the repository's ResNet18/CIFAR-10 benchmark cell is:

```bash
.venv/bin/python tasks/mal_sgdm_telemetry.py train \
  --output outputs/mal-sgdm-telemetry-resnet18-cifar10-seed42 \
  --device cuda --download --batch-size 256 --lr 0.1 --weight-decay 0.0005 \
  --epochs 200 --schedule cosine --warmup-epochs 5 \
  --amp-dtype bfloat16 --float32-precision tf32 --workers 4 \
  --wandb-mode online --wandb-entity osuwaidi-khalifa-university \
  --wandb-project MAL_benchmark
```

For paper evidence, `run-mal-sgdm-telemetry.sh` is submitted as the array
`1-6`: seeds 42, 1337 and 2026 under the canonical five-epoch warmup + cosine
recipe, paired with the same seeds under constant LR.  Each run has its own
lossless output directory beneath one suite directory.  The GPU array can be
throttled independently (for example, `--array=1-6%5`) without changing the
experimental design.

After all six tasks succeed, `run-mal-sgdm-telemetry-analysis.sh` runs on a CPU
node.  It validates the exact paired recipe and full tensor-step coverage, then
`analysis/analyze_mal_sgdm_telemetry_suite.py` creates tagged CSV source tables,
a descriptive report, and PNG (400 dpi), PDF and SVG versions of eight figures:
model-wide `q`/`c` evolution, depth evolution and scheduler difference,
parameter-kind distributions under both weighting views, stage threshold
occupancy, the complete tensor map, and the associated training/validation
curves.  Faint traces are individual seeds and aggregate bands are the observed
three-seed range; tensor steps are never presented as independent replicates.

The model and augmentation defaults follow `tasks/cifar_train.py`: a 3×3 stride-1 stem, no initial max-pooling, GroupNorm, a 10-class linear head, random crop/flip, RandAugment and random erasing. `--norm batch` switches to BatchNorm, and `--augmentation basic` selects just crop/flip and normalization. These choices affect gate dynamics, so compare runs with matching recipes. Batch size is 256. The official training set is split into 42,500 training and 7,500 validation images using a fixed stratified split; the indices are saved. Validation does not update the optimizer or enter telemetry. The official test set is untouched during fitting and model selection, then evaluated exactly twice: for the final model and for the checkpoint selected by validation accuracy.

Training defaults to float32 without clipping or gradient accumulation. `--amp-dtype bfloat16 --float32-precision tf32` matches the CUDA benchmark recipe. Seeds, source hashes, library versions, the normalization choice and optimizer groups are saved in `run.json`. Seeded runs are not guaranteed bitwise identical across devices. `--workers` defaults to zero for macOS portability; it can be increased for throughput. The incomplete last training batch is dropped, matching the existing CIFAR runner.

A short **synthetic pipeline check** needs no dataset or download:

```bash
.venv/bin/python tasks/mal_sgdm_telemetry.py train \
  --output analysis/mal_telemetry_smoke \
  --device mps --synthetic --batch-size 4 --epochs 2 --max-steps 6 \
  --flush-steps 2 --log-every 2
```

Synthetic results are explicitly labeled and cannot establish CIFAR-10 coefficient behavior. A real training run is needed to assess the 0.7 hypothesis.

## What is measured

For the default optimizer, let `m` denote the **previous stored heavy-ball buffer**, and `g` the gradient consumed by the optimizer:

```text
probe = g + 0.9 * m
s = clamp(sum(g * probe) / (max(norm(g), 1e-8) * max(norm(probe), 1e-8)), -1, 1)
q = (1 + s) / 2
c = 0.9 * q                    # beta_eff; falls back to 0.9 when norm(g) == 0
parameter -= lr * (g + c * m)
stored_buffer = probe          # in_place=False
```

The script uses an optional observer in the real `MAL_SGDM.step()` implementation. It does **not** reconstruct coefficients from a second optimizer or from momentum after the step. The observer receives the computed values before the update, and the recorder commits them only after `optimizer.step()` returns successfully. Enabling observation leaves the update arithmetic and checkpoint state unchanged.

Five scalars are recorded in this fixed feature order:

| Feature | Meaning |
|---|---|
| `cosine` | The clamped tensor-level cosine used by MAL before shifting and exponentiation |
| `gate_q` | The alignment gate `q`, before multiplying by beta or applying the zero-gradient fallback |
| `beta_eff` | The coefficient `c` actually multiplying the previous momentum in the update |
| `gradient_norm` | Norm of the optimizer's gradient, including coupled decay when enabled |
| `probe_norm` | Norm of the fixed-beta probe used to score alignment |

**The two coefficients have different scales:** under the defaults, `q=0.7` means `c=0.63`; `c=0.7` means `q≈0.7778`. The analysis shows **both** and uses 0.7 as a descriptive reference for each, not as an assumed universal steady state.

All original float32 coefficient bits are preserved with lossless compression. Norms use the optimizer's own dtype and epsilon floor. There is no sampling, rounding or lower-precision conversion. The callback performs no per-tensor host transfer; measurements are stacked and transferred once per shard. Recording still adds some allocation, synchronization and I/O overhead.

### Provenance and special cases

`tensor_metadata.json` and `.csv` map each tensor ID to its full parameter name, module class, parameter role, family (`conv`, `norm`, `linear`), detailed kind (`conv_weight`, `norm_bias`, etc.), shape, element count, optimizer group, effective weight decay and depth. The manifest follows `model.named_parameters()` order; recording uses tensor identities, so optimizer decay-group reordering cannot scramble provenance.

Depth is defined explicitly: stem = 0; the eight residual blocks `layer1.0` through `layer4.1` = 1–8; head = 9. Main and shortcut branches of a residual block share that depth; `branch` distinguishes them. `stage` also supports coarser aggregation over `layer1`–`layer4`. This is logical residual-block depth, not an ordering of every operation in the branched computation graph. The default model has 62 trainable tensors: 20 convolution weights, 20 norm weights, 20 norm biases and a linear weight/bias pair. Its convolutions have no biases. Running-mean/variance buffers in BatchNorm are not trainable parameters and are excluded.

- `grad=None`: no update or observation; all five stored features are NaN, `observed=False`, and that tensor's update counter does not advance.
- Zero gradient norm: the implementation returns a raw `q=0.5` for `pwr=1`, but **uses `c=0.9`** as its fallback. That `q` is preserved in raw data and excluded from alignment-gate summaries because alignment is undefined. The actual `c` is included in applied-coefficient summaries.
- Zero probe norm: alignment is also undefined. The raw values remain recorded; only `q` is excluded from alignment summaries. The applied coefficient remains whatever the optimizer actually used.
- Positive norms below `1e-8`: the optimizer's norm floor affects the score. These values remain in summaries and are counted under `epsilon_clamped_alignment`.
- Nonfinite coefficients are preserved in raw data, counted, and excluded from coefficient statistics. Nonfinite gradient/probe norms make alignment undefined. Counts are provided so exclusions are visible.

When decay is enabled, the optimizer excludes bias and other one-dimensional tensors from decay. Consequently, the gradient consumed by MAL can differ from `parameter.grad`; the manifest records effective decay per tensor.

## Files and analysis

| Output | Contents |
|---|---|
| `run.json` | Recipe, provenance, planned/completed steps, completion status |
| `tensor_metadata.json`, `.csv` | Stable tensor-ID mapping and classifications |
| `split_indices.npz` | CIFAR training/validation split indices |
| `telemetry/index.json` | Ordered list of atomically written telemetry shards |
| `telemetry/steps_*.npz` | Every recorded tensor on every completed step |
| `train_metrics.csv` | Training/validation loss and accuracy per epoch; partial epochs flagged |
| `source/` | Copies of the training script and optimizer source used for this run |
| `final_checkpoint.pt` | Final-epoch model and optimizer state for real CIFAR runs |
| `best_validation_checkpoint.pt` | Validation-selected model and its corresponding test metrics |
| `analysis/summary.csv` | Full-run and final-window statistics by model, tensor, kind, role, depth and stage |
| `analysis/epoch_summary.csv` | The same groupings by epoch, for evolution over training |
| `analysis/step_model_summary.csv` | Model-wide means/std at original step resolution, loss and learning rates |
| `analysis/histograms.csv` | Full/final-window distributions for every grouping, with 50 bins over [0, 1] |
| `analysis/epoch_model_histograms.csv` | Model-wide distribution evolution by epoch |
| `analysis/epoch_model_distributions.csv` | Exact epoch-level means and quantiles under both weighting views |
| `analysis/analysis.json` | Validity rules, thresholds, weighting and exact final-window step boundaries |
| `analysis/plots/` | Separate `gate_q`/`beta_eff` evolution, tensor heatmaps, means and occupancy figures |

Each NPZ shard has:

```text
values          [steps, tensors, 5]   # feature order above; tensor axis = manifest
observed        [steps, tensors]      # bool; distinguishes missing from zero
tensor_step     [steps, tensors]      # cumulative count of this tensor's updates
step            [steps]               # global optimizer step, starts at 1
epoch, batch    [steps]               # both start at 1; batch resets each epoch
batch_size      [steps]
samples_seen    [steps]               # cumulative training examples, including repeats
learning_rates  [steps, groups]        # rates used on these steps, not the next rates
loss            [steps]               # batch mean cross-entropy before the update
```

Every non-tensor summary is reported two ways. **Equal tensor** gives one vote per valid tensor-step observation, so a bias tensor and a large convolution tensor each contribute one coefficient. **Parameter-count weighted** weights that same tensor-level coefficient by the tensor's `numel`; it answers what a uniformly sampled scalar parameter experiences, but it does not pretend the repeated coefficient is an independent measurement for every element. Standard deviation describes pooled observations and is not a standard error assuming independent steps.

When W&B is enabled, only epoch-level training, validation, and compact telemetry aggregates are logged as scalar histories. The lossless shards, manifest, source snapshots, numeric analysis, and figures are uploaded together as an `optimizer-telemetry` artifact; there is no per-tensor-per-step scalar flood in the W&B history.

Threshold percentages use exactly these bins: **`x < 0.5`**, **`0.5 <= x <= 0.7`**, **`x > 0.7`**. Comparisons promote stored values to float64 and do not round them. For example, float32's representable value nearest 0.7 is slightly below decimal 0.7 and belongs to the middle bin. Percentages use valid observations as their denominator and sum to 100% when that denominator is nonzero. Histogram intervals are left-closed/right-open, except the final bin includes 1.

The default “steady-state estimate” is the mean over the **last 20% of persisted global optimizer steps**, rounded up to include a whole number of steps. It is labeled `late`, not a claim that the process has reached stationarity. This window includes a partial final epoch when present. Early self-alignment is included in full-run statistics; inspecting epoch curves and comparing final windows helps distinguish transient effects from sustained differences.

To change the window or regenerate analysis without retraining:

```bash
.venv/bin/python tasks/mal_sgdm_telemetry.py analyze \
  analysis/mal_sgdm_telemetry_cifar10_seed42 --late-fraction 0.1
```

`--no-plots` writes numeric outputs without requiring matplotlib. Plot generation uses the repository's development dependency on matplotlib.

Shards and their index are updated atomically every `--flush-steps` completed updates, at epoch boundaries, and on ordinary Python exceptions/keyboard interruption. An abrupt process kill or machine failure can lose the current in-memory chunk; previously indexed shards remain analyzable. Analysis reads a fixed snapshot of the index and validates global step continuity and per-tensor update counts. It can analyze an interrupted or still-running run's persisted prefix, which is labeled accordingly. It does not resume training. Use `--flush-steps 1` if persistence after every individual update is worth the extra overhead.

Read an individual tensor directly:

```python
import json
from pathlib import Path
import numpy as np

run = Path("analysis/mal_sgdm_telemetry_cifar10_seed42")
metadata = json.loads((run / "tensor_metadata.json").read_text())
tensor_id = next(row["tensor_id"] for row in metadata if row["name"] == "layer3.0.conv1.weight")
index = json.loads((run / "telemetry/index.json").read_text())
for entry in index["chunks"]:
    with np.load(run / "telemetry" / entry["file"], allow_pickle=False) as shard:
        steps = shard["step"]
        observed = shard["observed"][:, tensor_id]
        cosine = shard["values"][:, tensor_id, 0]
        q = shard["values"][:, tensor_id, 1]
        applied_coefficient = shard["values"][:, tensor_id, 2]
        # Pair these arrays with the manifest; do not infer identity from optimizer group order.
```

Focused numerical checks:

```bash
.venv/bin/python -m unittest checks.mal_sgdm_telemetry_checks checks.mal_optimizer_checks -v
```
