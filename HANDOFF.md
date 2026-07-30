# HANDOFF — Memory-ALigned (MAL) Momentum

Status snapshot: **2026-07-30**, branch `main` @ `f368ab3` (clean tree, plus one
untracked `diagnostics/` folder — see §3.4).

This document is written for a fresh Claude Code session on a different machine with
**no access to the originating conversation**. Everything load-bearing is recorded here,
including empirical results whose generating code no longer exists (§9).

---

## 1. Project objective and requirements

### 1.1 The idea

**Memory-ALigned (MAL) momentum**: a momentum optimizer whose *retention coefficient* is
modulated per-parameter-tensor by how well the accumulated momentum still agrees with the
fresh local gradient.

Rationale: heavy-ball momentum accelerates only while the descent direction stays
*consistent*. When the loss landscape's local geometry rotates or the optimizer bounces
across a valley, the momentum buffer becomes **stale** — it encodes an outdated direction
and actively drags the update the wrong way. MAL detects this via a cosine alignment
signal and decays the memory in proportion to the misalignment, instead of carrying a
fixed β.

### 1.2 Exact update rule (as implemented in `mal_sgd.py::MAL_SGD`)

Let `m` be the momentum buffer, `g` the current (L2-regularized) gradient, `β_probe` the
stored per-tensor coefficient.

```
m̂ = β_probe · m + g                      # probe the ORDINARY momentum candidate
r  = (1 + cos(m̂, g)) / 2                 # retention in [0, 1]; equivalently r = 1 − d,
                                          # with d = (1 − cos)/2 the normalized cosine distance
static   (adaptive=False):  c = β · r      # β is a group-level float constant
adaptive (adaptive=True):   c = r          # c is stored back as the next β_probe (per tensor)

m ← c · m + g                             # commit
p ← p − lr · m                            # plain
p ← p − lr · (g + c · m)                  # nesterov=True (Sutskever/PyTorch form, same c)
```

Properties worth preserving: static MAL reduces **exactly** to PyTorch SGD(M/N) when
`r == 1`; a fresh buffer (`m = 0`) makes the probe self-aligned so `r = 1` and the first
steps are un-damped (graceful warm-up); a zero gradient yields **no alignment evidence**
and the previous coefficient is retained (not an artificial 0.5).

### 1.3 Deliverable requirements

- Vision track: CIFAR-10 / CIFAR-100, ResNet-18/50 with **GroupNorm** (not BatchNorm),
  W&B sweeps over `align × nesterov × batch_size × lr × seed`.
- Transformer track (intended, **not yet wired**): ViT and LLM training via
  `MAL_ADAMW`, benchmarked against `CAUTIOUS_ADAMW` and stock AdamW.
- Baseline competitor: **Cautious Optimizers** (arXiv:2411.16085), implemented in
  `cautious_sgd.py`.
- Paper framing: MAL keeps momentum's acceleration **without** momentum's fragility.

---

## 2. Architecture and key design decisions

Every decision below was empirically contested; the recorded reason is *why the
alternative lost*. Do not "simplify" these without re-running the relevant ablation.

### 2.1 Probe on `m̂` (absorb-first), never on pre-absorption `m` — **critical**

The alignment cosine is computed between the *candidate update* `m̂ = β·m + g` and `g`,
**not** between the raw buffer `m` and `g`.

Why: consecutive SGD gradients are **negatively autocorrelated** (after `p −= lr·m` the
next gradient partly points back). Measuring `cos(m, g)` therefore reads ≤ 0 most of the
time under minibatch noise, collapsing adaptive β to ≈ **0.31** and destroying the memory.
Measured cost of the pre-absorption probe: **−6 to −7 accuracy points**, landing *below*
vanilla SGDM. The absorb-first probe operates at β ≈ 0.78–0.81.

Secondary benefits: `m̂`-probing is magnitude-aware (`cos(m̂,g) < 0` requires
`β⟨m,g⟩ < −‖g‖²`, so a small opposing memory is correctly ignored), and it judges the
object that actually gets applied.

### 2.2 Adaptive β is **uncapped** — do not add a 0.9 ceiling

`adaptive=True` stores `c = r ∈ [0, 1]`, which can exceed the nominal 0.9. Telemetry
(§9.4) shows β > 0.9 on **~33% of layer-steps**. A `β_max = 0.9` cap variant (`ada09`)
was tested: it wins on a small CNN but **loses at ResNet scale**. The uncapped rule is
self-correcting — β is recomputed from alignment every step, so a transient β ≈ 1.0 is
revoked as soon as misalignment appears (unlike a fixed β = 1, which is an undamped
integrator).

### 2.3 `MAL_ADAMW` gates only the β₁ EMA coefficient

- The gate modulates **only** the first moment. The second moment `v` (a scale tracker)
  and its bias correction are untouched standard AdamW.
- EMA form `m ← c·m + (1−c)·g` keeps **unit mass**, so a dynamic `c` cannot distort the
  update magnitude — the concern that exists in the SGD variant is structurally absent.
- **Exact bias correction under dynamic c**: a per-parameter running product
  `bc_prod = Π c_s` is tracked and the correction is `1 − bc_prod`, which provably reduces
  to `1 − β₁ᵗ` for constant `c` (verified: matches `0.9³⁰` to 1e-6).
- `MAX_BETA1 = 1 − 1e-4` cap **is** required here (unlike §2.2): in EMA form `c = 1` puts
  zero mass on `g` (frozen memory, zero bias correction, 0/0 on step 1). With the cap, a
  perfectly-aligned first step reproduces plain AdamW's first step exactly.
- Weight decay is **decoupled** and applied outside the gate, so the alignment signal sees
  only the true gradient.
- 1-D params (LayerNorm gains, biases) keep a **fixed** β₁ unless `align_1d=True`
  (cosine on small tensors is noise-dominated). Note the *SGD* evidence points the other
  way (§9.3) — treat `align_1d` as an untested ablation knob, not a settled rule.

### 2.4 Weight decay conventions

- `MAL_SGD` / `CAUTIOUS_SGD`: **coupled** L2, folded into `g` **out-of-place**
  (`g = g.add(p, alpha=wd)`) so `p.grad` is never mutated.
- Both AdamW classes: **decoupled** (`p.mul_(1 − lr·wd)`).
- Biases and 1-D norm params are excluded from decay at construction time (they land in a
  `weight_decay=0.0` param group).
- Consequence to be aware of: with coupled decay the term `wd·p` enters the alignment
  signal. Late in training (small true gradients) it is a slowly-varying, self-consistent
  direction that pushes `d` down and β *up*. Part of any late-training β rise is
  wd-driven, not pure signal.

### 2.5 GPU-sync-free by construction

The adaptive per-tensor β is stored as a **0-dim CUDA tensor** (`p.new_tensor(...)`), and
the probe uses `torch.addcmul(g, m, beta_tensor)`. Never:
- pass a 0-dim tensor as `alpha=` (it is a *Scalar* arg → forces an implicit `.item()`
  → one host sync per parameter per step; this regression was measured at **+47% step
  time** on MPS),
- call `.item()` per layer for logging. Stack all betas into one 1-D tensor and do a
  single `.cpu()`.

### 2.6 AMP / mixed precision (in `main.py`, **not** in the optimizers)

The optimizers are deliberately amp-agnostic: they run under `@torch.no_grad()` outside
`autocast`, and all state (params, `momentum`, `beta`, `v`, `bc_prod`) is **fp32**. Low
precision is confined to forward/backward. The alignment math — the one thing that must
not be fp16 — is therefore always fp32. **No amp changes belong in the optimizer files.**

`main.py` specifics:
- `GradScaler` is enabled **only** for fp16 (`enabled=amp_enabled and dtype==float16`);
  bf16 has fp32 exponent range and needs no loss scaling.
- Skipped-step detection: `scaler.get_scale() >= scale_before_step` gates both the
  `optimizer_step` counter and `lr_scheduler.step()`, so an inf-gradient step does not
  advance the schedule.
- TF32 via the post-2.9 API: `torch.backends.cuda.matmul.fp32_precision` and
  `torch.backends.cudnn.conv.fp32_precision`.
- **bfloat16 is the default** `--amp_dtype` (Ampere+). Use fp16 only on pre-Ampere
  (V100/T4); a `torch.cuda.is_bf16_supported()` guard raises a clear error.

---

## 3. Files

### 3.1 Core implementation (tracked, on `main`)

| File | Contents |
|---|---|
| `mal_sgd.py` | `_get_cosine_sim()` helper; `MAL_SGD` (adaptive/static × plain/nesterov); `MAL_ADAMW` (adaptive/static, `align_1d` flag) |
| `cautious_sgd.py` | `CAUTIOUS_SGD` (± nesterov); `CAUTIOUS_ADAMW` — the arXiv:2411.16085 baseline |
| `main.py` | Single training entry point (CIFAR-10/100, ResNet-50/timm, GN, warmup+cosine, AMP, W&B). Reads `align`, `nesterov`, `batch_size`, `lr`, `seed` from the sweep config |
| `create_sweep.py` | Builds the W&B sweep: `align ∈ {MAL, MAL_ada, none, cautious}`, `nesterov ∈ {T,F}`, 7 batch sizes, 8 lrs, 3 seeds |
| `curved_ravine_demo.py` | Self-contained 2-D motivation demo (see §3.3) |
| `sync-venv.sh`, `wb-agents.sh` | Cluster venv builder + SLURM sweep-agent launcher (see §5.3) |
| `pyproject.toml` | Python 3.14, torch 2.11.0 pinned to the **cu128** index |
| `figures/` | `curved_ravine_{objective,trajectories}.{png,svg}`, `curved_ravine_metrics.json`, `heatmap plot.png` |

### 3.2 Branches

- **`main`** @ `f368ab3` — the live branch. All work should continue here.
- **`per_output_mal`** @ `9bca4d2` — per-output-unit granularity experiment.
  ⚠️ **STALE**: it branched *before* the `mal_sgd.py` rewrite, so it reverts
  `_get_cosine_sim`, the docstrings, and the dual-beta-layout refactor. Do **not** merge
  it. Re-implement the idea on top of current `main` (§8, item 2).
- `claude/mem-align-momentum-scaling-92ee6b` — old session worktree branch; ignore/delete.
- Two stale git worktrees exist under `.claude/worktrees/`; they hold old copies of
  `mal_sgd.py`. **Always edit `/Users/omar/Python/mem_align/mal_sgd.py`** — attachments and
  greps that resolve to a worktree path are showing pre-rewrite code.

### 3.3 `curved_ravine_demo.py` (the motivation figure)

A rigorous version of the "momentum harm" demonstration, with a *smooth, single-valued*
sinusoidal ravine rather than a polar spiral (no angular singularity, one centerline):

```
F(x,y) = axial/2 · x² + wall/2 · q(x,y)² ,  q(x,y) = y − amplitude·sin(frequency·x)
```

Unique global minimum at the origin; valley centerline `q = 0`; non-convexity is
*certified* by sampling Hessian eigenvalues (min eigenvalue ≈ −75, negative-curvature
fraction ≈ 0.49 over 77k samples). Committed results (`figures/curved_ravine_metrics.json`,
lr 0.02, β 0.9, 25 steps): **GD** final 0.923 (0 uphill steps) · **GDM** final 2.425
(11 uphill steps, 44%, min update-gradient cosine −0.995) · **MAL** final **1.07e-05**
(4 uphill steps, median effective momentum 0.777, min 0.087). This is the cleanest
existing artifact of the core claim.

### 3.4 `diagnostics/` — **UNTRACKED, must be committed to survive**

Created at handoff time by copying the last surviving scratchpad work:

- `beta_diag.py` — trains ResNet-18-GN on a CIFAR-10 subset with `adaptive=True` and
  snapshots **every per-tensor β at every step** to `beta_telemetry.npz`. Currently
  `DEVICE = "mps"`; **change to `"cuda"`** to use on the cluster.
- `analyze_beta.py` — produces the 5-panel telemetry figure + printed statistics.
- `beta_telemetry.npz` — the recorded run (800 steps × 62 tensors) backing §9.4.
- `beta_telemetry.png` — the rendered figure.

```bash
git add diagnostics && git commit -m "add adaptive-beta telemetry diagnostics"
```

Without this commit these files are lost — they live in a temp directory.

---

## 4. Current implementation status

**All nine optimizer configurations were functionally verified at handoff time** (60-step
MLP+LayerNorm training run, loss decreasing, all params finite):

| Configuration | Status |
|---|---|
| `MAL_SGD` adaptive / static / ±nesterov (4) | ✅ working |
| `MAL_ADAMW` adaptive / static (2) | ✅ working |
| `CAUTIOUS_SGD` ± nesterov (2) | ✅ working |
| `CAUTIOUS_ADAMW` (1) | ✅ working |
| `state_dict()` → `load_state_dict()` round-trip, same device | ✅ both AdamW classes |
| `MAL_ADAMW` per-parameter step counters | ✅ fixed in `c7297d8` |

Wiring status:
- ✅ `main.py` wires `MAL_SGD` (`align ∈ {MAL, MAL_ada}`), `torch.optim.SGD`
  (`align="none"`), `CAUTIOUS_SGD` (`align="cautious"`).
- ❌ **`main.py` does not wire `MAL_ADAMW` or `CAUTIOUS_ADAMW`** — the transformer track
  is implemented but unreachable from the training script.
- ✅ AMP path complete (bf16 default, fp16 scaler, TF32, skip-aware scheduler).
- 🔄 A CIFAR-100 sweep exists on W&B: `osuwaidi-khalifa-university/FINAL_MAL_CIFAR100/ecyt5b2h`
  (hardcoded in `wb-agents.sh`). Status of its results is **unknown to this document** —
  check W&B first thing.

---

## 5. Commands

### 5.1 Local (workstation / Mac)

```bash
cd /Users/omar/Python/mem_align
uv sync --all-groups
```

⚠️ **This fails on macOS.** The cu128 pin (§6) has no darwin wheels, so `uv sync` / `uv run`
in this project abort with *"doesn't have a source distribution or wheel for the current
platform"*. The repo env is **Linux/CUDA-only by construction**. To do CPU/MPS prototyping on
a Mac, either use a separate throwaway venv with stock PyPI torch, or add
`tool.uv.required-environments` / a darwin-conditional torch source to `pyproject.toml`.
Historical local MPS work ran via `uv run --project <a separate worktree with a Mac venv>`.

Ad-hoc script (note: `python` alone is not on PATH; always go through `uv run`):

```bash
uv run python curved_ravine_demo.py
```

Regenerate β telemetry (edit `DEVICE` in the script for cuda vs mps):

```bash
uv run python diagnostics/beta_diag.py     # writes beta_telemetry.npz
uv run --with matplotlib python diagnostics/analyze_beta.py
```

`matplotlib` is **not** a project dependency — inject it with `uv run --with matplotlib`
(or add it to the dev group).

### 5.2 Creating a sweep

```bash
uv run create_sweep.py main.py \
  --data cifar100 \
  --sweep_name "<name>" \
  --project_name "<wandb-project>" \
  --method grid
# prints: uv run wandb agent --forward-signals <entity>/<project>/<sweep_id>
```

Single manual run (sweep params must be supplied because `main.py` reads them from
`run.config`):

```bash
uv run python main.py --data cifar10 --arch resnet50 --epochs 200 --amp_dtype bfloat16
```

### 5.3 Cluster (SLURM) — read `sync-venv.sh` before touching the venv

Project root on the cluster: `/shared/b00090279/memory_align`.

```bash
./sync-venv.sh              # build venv for current uv.lock, flip .venv symlink
./sync-venv.sh --rebuild    # force rebuild
./sync-venv.sh --prune      # delete unreferenced old venvs (only when no jobs queued)
sbatch wb-agents.sh         # launch a sweep agent (1 GPU, 4 CPUs, 8G)
```

The venv scheme is **immutable + versioned by `uv.lock` hash**: `.venv-<hash>/` dirs are
`chmod a-w` after build, `.venv` is a symlink that gets atomically flipped, and running
jobs resolve the symlink once at start. This exists because an in-place `uv sync` on shared
NFS previously injected a mismatched CUDA stack and broke cuDNN on **every** node
(`CUDNN_STATUS_NOT_INITIALIZED`). Both scripts preflight with a **real convolution**, not
an import check, because a poisoned venv can import torch and report
`cuda.is_available() == True` while every conv fails — silently burning sweep runs.

### 5.4 Tests

There is **no test suite in the repo.** Regression tests written during development were
lost with a temp directory (§9.6). Recreating them is a priority task (§8, item 3). The
verification pattern that was used:

- static `MAL_SGD` with `r == 1` must match `torch.optim.SGD` (incl. `nesterov=True`)
  bit-for-bit — set a momentum buffer to `2·g` and compare a single step;
- `MAL_ADAMW` on 1-D-only params with `align_1d=False` must match `torch.optim.AdamW`
  exactly over ≥30 steps;
- `bc_prod` after N constant-`c` steps must equal `β₁ᴺ`;
- `CAUTIOUS_*` with a fully-contradicted update must leave `p` unchanged while the buffer
  still absorbs (mask applies to the update, never the state).

---

## 6. Assumptions and constraints

- **Python 3.14**, `torch==2.11.0`, `torchvision==0.26.0`, uv-managed.
- **CUDA 12.8 only.** The cluster driver is `570.86.15` (= CUDA 12.8). Default PyPI torch
  wheels are CUDA-13 builds and abort with *"driver is too old"*. `pyproject.toml` pins
  torch/torchvision to the `pytorch-cu128` index, and `sync-venv.sh` **refuses to build**
  if `uv.lock` contains CUDA-13 wheels or lacks a `+cu128` torch. Never regenerate the
  lock without that pin.
- Training targets CUDA exclusively — `main.py` hardcodes `DEVICE = torch.device("cuda")`.
  MPS was used only for local prototyping (via separate scripts, not `main.py`).
- **GroupNorm, not BatchNorm** (`num_groups = min(32, C//4)`), with a CIFAR stem
  (3×3 stride-1 conv, `maxpool = Identity`). GN was chosen so the batch-size axis of the
  sweep is not confounded by BN's batch-statistics dependence.
- Data split: 85% train / 15% val from the train set (stratified, seeded); the official
  test set is evaluated **once**, on the best-val checkpoint.
- Local dataset copies exist at `/Users/omar/Python/datasets/` (CIFAR10, CIFAR100, MNIST,
  FashionMNIST). Prefer them — the Toronto CIFAR mirror downloads at ~30 kB/s.
- Sweep grid is large (4 align × 2 nesterov × 7 bs × 8 lr × 3 seeds = 1344 runs). Use
  `--method random`/`bayes`, or prune axes, if compute-bound.

---

## 7. Known bugs, gotchas, and pitfalls

**Open issues:**

1. **AdamW checkpoint resume is not device-portable.** All per-parameter state
   (`m`, `v`, `beta`, `bc_prod`, `step`) lives in `param_groups`, **not** in
   `self.state`. PyTorch's `load_state_dict` only device-casts entries in `self.state`, so
   save-on-GPU → `map_location="cpu"` load → resume **crashes** with a device mismatch
   (confirmed). Same-device save/load works. Fix = move per-param state into
   `self.state[p]` (a real refactor; affects all four classes).
2. **`main.py` does not wire the AdamW classes** (§4).
3. **`--nesterov` argparse flag is dead.** `main.py` takes `nest` from `config.nesterov`
   (the sweep). The CLI flag has no `type=`, so `--nesterov False` would yield the
   **truthy string** `"False"`. Delete the flag or give it a proper bool parser.
4. **fp16 slightly under-runs the LR schedule.** `total_steps` assumes no skipped steps,
   but the fp16 scaler skips a few during calibration, so cosine never quite reaches
   `eta_min`. Harmless; nonexistent with bf16 (no scaler → no skips).
5. **Verify TF32 actually applied.** Confirm
   `torch.backends.cuda.matmul.fp32_precision` exists on the installed torch — assigning
   an unknown attribute on that object can silently no-op, losing TF32 without error.
6. **Cautious rescale shrinks small tensors.** `scale = numel / (mask.sum() + 1)` gives
   ×10/11 ≈ 0.91 even at *full* agreement for a 10-element bias, and is unbounded when the
   mask is nearly empty. The official implementation clamps the mask mean at `1e-3`. If
   you adopt `scale = numel / mask.sum().clamp_min(1e-3·numel)`, change **all three**
   call sites in one commit so SGD/NAG/AdamW comparisons stay consistent.
7. `scaler.unscale_(opt)` in `main.py` is **redundant** (`scaler.step` unscales
   internally). Harmless; the adjacent comment overstates its necessity.

**Historical bugs — already fixed, do not reintroduce:**

- `p.subcmul_(...)` — **does not exist** in PyTorch. Use `p.addcmul_(m, c, value=-lr)`.
- `Tensor.norm(dim=<3-tuple>)` dispatches to `matrix_norm` and **raises**
  (`dim must be a 2-tuple`). For block norms over arbitrary axes use
  `torch.linalg.vector_norm(x, dim=dims, keepdim=True)`.
- `torch.add(g, m, alpha=<0-dim tensor>)` forces a host sync per parameter (§2.5). Use
  `torch.addcmul`.
- `"beta": [tensor] * n` — list-multiply **aliases one object** across all layers. Use a
  comprehension: `[p.new_tensor(x) for p in params]`.
- `MAL_ADAMW` read `group["step"]` that `__init__` never created → `KeyError: 'step'` on
  the first call (fixed in `c7297d8`).
- Indexing a group-level float `beta` with `betas[i]` when `adaptive=False` →
  `TypeError: 'float' object is not subscriptable`. The two layouts are deliberate:
  per-param tensor list when adaptive, group float when static.

---

## 8. Remaining tasks, in priority order

1. **Check the running CIFAR-100 sweep** (`FINAL_MAL_CIFAR100/ecyt5b2h`) and decide
   whether it is answering the right question. The decisive comparisons are
   `MAL_ada` vs `none` across the **batch-size axis** and against a **tuned static-β**
   control (§9.5).
2. **Add a static β ≈ 0.80 SGDM control arm.** Telemetry shows adaptive β self-regulates
   to a mean of 0.806, so β = 0.9 is *not* the right null hypothesis. Without this arm a
   reviewer will ask whether MAL is just "SGDM at a better β". Cheap and high-value.
3. **Recreate the regression test suite** (§5.4) as a tracked `tests/` folder, and add a
   one-step smoke test for every optimizer class — a whole optimizer shipped
   dead-on-arrival once (`KeyError: 'step'`).
4. **Wire `MAL_ADAMW` + `CAUTIOUS_ADAMW` into `main.py`** (e.g.
   `align ∈ {MAL_adamw, MAL_ada_adamw, cautious_adamw, adamw}`), then validate on a ViT
   (timm) CIFAR run before attempting an LLM. Use `betas=(0.9, 0.95)` for LLM pretraining.
5. **Re-implement per-output-unit granularity on current `main`** (do not merge the stale
   `per_output_mal` branch). This produced the **best numbers of the whole project** in a
   preliminary test (§9.3). Implementation sketch, verified working:

   ```python
   if g.ndim > 1:
       dims = tuple(range(1, g.ndim))     # Linear (out,in)→(1,); ConvNd→(1..N-1)
       denom = (torch.linalg.vector_norm(m_hat, dim=dims, keepdim=True)
                * torch.linalg.vector_norm(g, dim=dims, keepdim=True)).clamp_min(1e-8)
       cos = ((m_hat * g).sum(dims, keepdim=True) / denom).clamp(-1.0, 1.0)
   else:
       # 1-D params (biases, GN/LN affines): keep the existing WHOLE-TENSOR cosine.
       # Per-axis on a 1-D tensor degenerates to per-coordinate sign gating, which is
       # the `hard_per` failure mode (§9.2).
       cos, _ = _get_cosine_sim(m_hat, g)
   ```
   `c` becomes `(out, 1, …, 1)` and broadcasts through `m.mul_(c).add_(g)`, the
   `addcmul` probe, and the NAG line. The constructor needs **no** change: the 0-dim
   `p.new_tensor(0.0)` init broadcasts on step 1, and `betas[i] = c` adopts the per-unit
   shape thereafter. ⚠️ Per-axis **helps adaptive and hurts static** — the stored β acts
   as a temporal filter on per-unit cosine noise; a one-shot multiplicative shrink
   compounds it instead. Ship it for `adaptive=True` only.
6. **Move per-param state into `self.state`** to fix device-portable checkpointing (§7.1)
   — required before any long/preemptible LLM run.
7. **β telemetry at scale**: port `diagnostics/beta_diag.py` to CUDA and log to W&B
   (§10.2). Two unexplored axes: does β drift with a real warmup+cosine schedule, and
   does β rise with batch size (the noise-adaptivity prediction)?
8. Optional/low priority: the descent-cap flag (§9.2, last row) purely as scaffolding for
   a convergence theorem; `align_1d=True` ablation for the AdamW track.

---

## 9. Empirical results — **the code that produced these no longer exists**

All MPS prototype harnesses, figure generators, and test suites were written to a
session temp directory that has since been cleaned. The **conclusions** are preserved
here; re-derive the code from these specs if the numbers need reproducing. Protocol
unless stated: small 4-conv+GN CNN (~250k params), CIFAR-10 10k-subset, bs 128,
wd 5e-4, **constant** lr, 15 epochs, mean of 2 seeds, best val acc.

### 9.1 Variant naming used below

`smooth` = `MAL_SGD(adaptive=False)` · `smooth_ada` / `ada` = `MAL_SGD(adaptive=True)` ·
`ada09` = adaptive with `c = 0.9·r` · `hard` = zero the buffer when `⟨m,g⟩ < 0` ·
`hard_per` = same per-coordinate · `sgd_b072` = plain SGDM at β = 0.72.

### 9.2 Rejected variants (all removed from the codebase — **do not resurrect**)

| Variant | Why it was rejected |
|---|---|
| `rect` — shrink only when `cos < 0` (`c = β·min(1, 1+cos)`) | Won **zero** axes. Its trigger almost never fires under minibatch noise, so it barely improves on vanilla SGDM (55.65 / 46.90 at lr 0.1 / 0.4). |
| `proj` — halfspace projection of `m` off `g` (gives an unconditional descent guarantee) | Won zero axes stochastically **and diverges deterministically below vanilla heavy-ball's stability edge**: above the plain-GD edge the anti-aligned memory is *load-bearing* (it cancels the growing oscillation), and deleting it degrades the steep direction to unstable plain GD. "Bounce ⇒ memory is wrong" is false in the linear regime. |
| `hard` (the original zero-reset ancestor) | ≈ equivalent to a static β = 0.72 control. Bounce rate 15% of layer-steps at bs 32 → 3% at bs 512 — restarts too often under noise (the SRSGD failure mode). Only competitive at large batch. |
| `hard_per` (per-coordinate reset) | **Catastrophic**: resets 26–31% of *all* momentum coordinates every step, so memory never accumulates. −19 points at ResNet scale (63.17 vs ~82). Contrast Cautious, which masks the *update* and leaves state intact. |
| `ada09` (β_max = 0.9 cap) | Best on the small CNN (58.85) but **loses at ResNet scale**; forfeits the β > 0.9 band (§2.2). |
| Magnitude-aware descent cap: `c ← min(c_raw, (1−ε)‖g‖²/(−⟨m,g⟩))` when `⟨m,g⟩ < 0` | **Behaviorally a no-op.** `⟨m,g⟩ < 0` on **60–90%** of layer-steps, yet the cap **binds on only 0.0–0.7%** — the absorb-first gate already shrinks harder exactly where the cap would act (at the descent boundary the gate keeps `β/2` vs the cap's `0.9β`). Deterministic probes bit-identical; all accuracy deltas within seed noise. Value is theoretical only: it upgrades "descends on ≥99.3% of steps" to "provably always". |

### 9.3 Vision results

**Tier A — lr stress (small CNN), mean best val acc:**

| variant | lr 0.05 | 0.1 | 0.2 | 0.4 | 0.8 | 20% label noise @0.1 |
|---|---|---|---|---|---|---|
| sgd (β=.9) | 56.62 | 54.15 | 50.88 | 43.40 | 31.73 | 48.42 |
| sgd (β=.72 control) | 56.25 | 56.38 | 55.55 | 51.42 | 44.40 | 50.80 |
| smooth | 57.15 | 58.50 | 57.27 | 54.55 | 49.08 | 51.40 |
| **smooth_ada** | 58.50 | 58.12 | 57.08 | **55.15** | 48.55 | **52.62** |
| ada09 | 57.08 | **58.85** | **57.33** | 54.85 | **49.62** | 51.05 |
| hard | 56.80 | 55.60 | 53.92 | 50.17 | 44.42 | 49.42 |
| hard_per | 51.40 | 51.00 | 48.73 | 44.70 | 41.95 | 48.00 |

The smooth family sweeps every column **and** beats the static β = 0.72 control at every
lr — so MAL is *not* reducible to "SGDM with a smaller β".

**Tier B — batch size (best over lr):** bs 32 / 128 / 512 →
smooth_ada **57.70** / **58.50** / 50.35 · ada09 57.17 / **58.85** / 51.22 ·
smooth 56.67 / 58.50 / 49.67 · hard 57.08 / 56.80 / **51.95** · sgd 56.33 / 56.62 / 51.05.
The ranking **flips with gradient SNR**: the smooth family wins where gradients are noisy;
at bs 512 `hard` wins and plain `smooth` drops *below* vanilla SGD — the persistent shrink
is a real tax when gradients are clean.

**Tier C — realistic protocol** (ResNet-18-GN, repo recipe, warmup+cosine, crop/flip aug,
20k train, bs 128, 20 epochs), mean best val acc at lr 0.1 / 0.4:
sgd **82.84** / 80.44 · **smooth_ada 82.11 / 81.07** · hard 82.16 / 79.30 ·
smooth 80.15 / 80.30 · ada09 80.56 / 80.15 · sgd_b072 79.81 / 80.56 · hard_per 63.17 / —.

Key honest finding: **with a real schedule + augmentation, tuned vanilla SGD is strong**
(the schedule absorbs much of the high-lr fragility MAL exploits at constant lr).
`smooth_ada` is the only variant that matches SGD's peak *and* wins at the stress lr with
the best worst-case. Frame the contribution as **"a much wider sweet spot"**, not "higher
peak accuracy".

**Per-output-unit granularity (preliminary, small CNN):** `ada_axis` **58.75 / 55.90**
(lr 0.1 / 0.4) vs per-tensor `ada` 57.95 / 54.78 — the largest single improvement observed
in the project. But `smooth_axis` (static) **regressed** to 57.20 / 52.92. Per-unit β
spread is large and real (std 0.18–0.32 across filters in one layer; the CIFAR stem conv
with fan-in 27 is noisiest, min β 0.04). **Not yet validated at Tier-C scale.**

### 9.4 Adaptive-β telemetry (ResNet-18-GN, CIFAR-10, 800 steps × 62 tensors, lr 0.1 constant)

Raw data: `diagnostics/beta_telemetry.npz`; figure: `diagnostics/beta_telemetry.png`.

| metric | all steps | steady-state (last 50%) |
|---|---|---|
| mean β | 0.801 | 0.806 |
| max / min β | 1.000 / 0.009 | 1.000 / 0.041 |
| **% β > 0.9** | **34.1%** | 33.3% |
| **% β < 0.1** | 0.1% | 0.1% |

By tensor type (steady-state mean, % > 0.9): weight 0.809 / 24.6% · bias 0.804 / 38.3% ·
norm-gain 0.806 / 37.2%. Mean β is **remarkably uniform across depth and size**
(0.79–0.81, no input→output gradient); what differs is *volatility* —
`corr(β, log numel) = +0.37` is a variance effect (small 1-D tensors have noisier cosines
and swing into both extremes), not a mean shift. The first ~15 steps are pinned at β = 1.0
(zero buffer ⇒ self-aligned probe by construction).

Actionable: (a) the honest static control is **β ≈ 0.80**; (b) hard resets essentially
never fire (0.1% below 0.1) — the gate does continuous gentle modulation, so do **not**
design bounce-triggered logic around this regime; (c) the β > 0.9 band is real and used a
third of the time.

### 9.5 Momentum-harm landscape study (for the paper's motivation section)

Honest bracketing, established by measuring **distributions over random starts**, never a
single trajectory (single 2-D trajectories in these landscapes are chaotically
start-sensitive — an early conclusion had to be reversed after checking distributions):

- **Momentum HELPS** on a smooth ill-conditioned quadratic (accelerates the low-curvature
  axis; its stable-lr window is *wider* than GD's, per `2(1+β)/L` vs `2/L`).
- **Momentum HELPS** on a *rugged downhill funnel* (`½‖x‖² + A·Σ(1−cos(ωxᵢ))`) — inertia
  escapes the local minima that trap GD, in **85% of random starts**. ⚠️ The original
  hypothesis (that rugged funnels would show momentum harm) is **false**; do not use this
  landscape as a harm example.
- **Momentum HARMS** on a *meandering / rotating* valley: worse than GD in **95% of
  starts** (median ~300×). This is the real harm mechanism → `curved_ravine_demo.py`.
- **Momentum HARMS** via the **stochastic noise floor**: on an isotropic bowl *with no
  local minima*, matched-lr momentum inflates the stationary loss monotonically with β
  (up to **100×** at β = 0.99), and a matched-*effective*-lr control (`lr·(1−β)`) is
  perfectly flat — proving momentum provides **zero** variance benefit, only effective-step
  amplification. MAL cut this penalty ~3×. **This is the regime that matters for NN
  training.**
- Why momentum is *wigglier yet faster* (asked and answered): on the bottleneck axis GD has
  a real eigenvalue (+0.98, overdamped, monotone crawl) while heavy-ball has **complex**
  eigenvalues (0.94 ± 0.13i, underdamped → overshoots, 13 zero-crossings measured) with a
  **smaller modulus** (0.949 < 0.98). Overshoot and speed are the *same* phenomenon. In a
  rotating valley each overshoot lands on a *new uphill* wall, so the ringing amplifies
  instead of decaying → divergence. MAL cuts β exactly then.

### 9.6 Lost artifacts (rebuild only if needed)

MPS bake-off harness (`bench.py` + tiered grids), the spiral/quadratic/Rosenbrock toy
objectives and figure generators, the underdamping explainer figure, `axis_mal.py`,
`cap_variants.py`, `ada09.py`, and the four regression test scripts. `curved_ravine_demo.py`
(tracked) supersedes the toy-figure work; §5.4 specifies the tests to rebuild.

---

## 10. Conversation context not evident from the code

1. **Working style.** Keep optimizer lineages separated per branch rather than building one
   omnibus implementation with many flags — each branch is a controlled experiment, and
   conflating them muddies attribution. Write a math-backed, literature-checked,
   falsifiable hypothesis *before* running comparisons; validate with small local
   prototypes before committing cluster time. Report distributions, not single runs.
2. **W&B logging pitfall.** The scripts previously logged with `run.log(..., step=epoch)`.
   W&B requires monotonically increasing steps, so mixing explicit `step=` with any
   step-level (per-iteration) logging **silently drops** the epoch-level metrics. `main.py`
   now logs `epoch` as a normal key. If you add per-step β telemetry, use the
   `define_metric` pattern (commented-out scaffolding already exists in
   `main.py::train_val_model`):
   ```python
   run.define_metric("epoch"); run.define_metric("train_loss", step_metric="epoch")
   run.define_metric("optimizer_step"); run.define_metric("mal_beta/*", step_metric="optimizer_step")
   ```
   Log β as one `wandb.Histogram` plus scalars (`beta/mean`, `beta/p05`, `beta/p95`,
   `beta/frac_gt_0.9`) every N steps — one device→host copy of a ~62-float vector,
   negligible overhead.
3. **Positioning / related work** (for the paper's related-work section). MAL's novelty
   claim is that it **edits the momentum state itself**; the neighbors modulate other
   things:
   - *Adaptive restart* (O'Donoghue & Candès, arXiv:1204.3982) and **SRSGD**
     (arXiv:2002.10583) reset momentum to zero — the `hard` ancestor. SRSGD's key finding
     (gradient-based restarts fire too often under stochastic gradients and degrade toward
     momentum-free SGD) is exactly what the smooth gate is designed to avoid, and it
     matches the measured 15%-bounce-rate at bs 32.
   - **Cautious Optimizers** (arXiv:2411.16085) mask the *update* per-coordinate; their
     convergence argument rests on the masked update keeping a positive inner product with
     `g`. This is the repo's implemented baseline.
   - **PCGrad** (arXiv:2001.06782) projects conflicting *task* gradients; the rejected
     `proj` gate was essentially PCGrad applied temporally.
   - Nearest neighbors that use `cos(m, g)` but modulate something else: **Hindsight-Guided
     Momentum** (arXiv:2506.22479, modulates the *learning rate*), **Adaptive Braking**
     (arXiv:2007.01397, scales the *gradient*), and projection-discriminant Adam
     (arXiv:2503.10005). None edit the momentum buffer.
4. **Reviewer objections to pre-empt.** (a) "Is this just a smaller β?" → the β = 0.72/0.80
   control arms answer it. (b) "Why probe `m̂` and not `m`?" → §2.1 is a paper-worthy
   ablation row. (c) "Momentum only looks bad because your lr is too high" → the correct
   framing is that a rotating/noisy landscape *shrinks momentum's stable-lr window*, and
   MAL restores it **without** sacrificing speed (it was fastest at its own optimal lr on
   every landscape tested).
5. **The user edits this repo concurrently and has rewritten branch history mid-session.**
   Always re-check `git log` / `git status` and re-read files immediately before editing;
   do not trust a file view from earlier in a session. Beware attachments that resolve to
   `.claude/worktrees/...` — those are pre-rewrite copies.
6. **Negative results are deliberately documented, not deleted** (§9.2). They are
   defensible paper content ("we tried the principled descent-guarantee variant; it is a
   measured no-op because the gate already dominates it") and prevent re-litigating
   settled questions.

---

## 11. CUDA / GPU work to perform next

Ordered roughly by risk-reduction value. Items 1–3 are environment validation and should be
done **before** trusting any new sweep result.

### 11.1 Validate the environment (do this first, on a compute node)

```bash
python -c "
import torch, torch.nn as nn
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| cudnn', torch.backends.cudnn.version())
print('device', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))
print('bf16 supported:', torch.cuda.is_bf16_supported())
nn.Conv2d(3,16,3,padding=1).cuda()(torch.randn(2,3,32,32,device='cuda')); torch.cuda.synchronize()
print('real conv OK')
"
```
Expect `cuda 12.8` and a `+cu128` torch build (§6). A conv that *runs* is the only valid
health check — an import check passes on a venv whose cuDNN cannot initialize.

**Confirm TF32 actually applied** (§7.5) — a silent no-op here costs ~2× matmul throughput:
```bash
python -c "
import torch
torch.backends.cuda.matmul.fp32_precision='tf32'; torch.backends.cudnn.conv.fp32_precision='tf32'
print('matmul:', torch.backends.cuda.matmul.fp32_precision, '| conv:', torch.backends.cudnn.conv.fp32_precision)
"
```
If either attribute is missing on torch 2.11, fall back to
`torch.backends.cuda.matmul.allow_tf32 = True` / `torch.backends.cudnn.allow_tf32 = True`.

### 11.2 Prove the optimizer is still sync-free on CUDA

The §2.5 property was measured on MPS. Verify on CUDA with PyTorch's built-in detector —
any hidden device→host sync inside `step()` will raise:

```python
torch.cuda.set_sync_debug_mode("error")   # or "warn"
# ...run ~20 training steps of each optimizer variant...
torch.cuda.set_sync_debug_mode("default")
```
Note `loss.item()` in the training loop is an intentional sync — call it outside the
guarded region, or use `"warn"` and read the stack traces. Adaptive `MAL_SGD` and both
AdamW classes are the ones to check (they carry per-tensor tensor state).

### 11.3 Establish the per-step overhead budget

`step()` is a **Python loop over ~62 parameter tensors**, each issuing ~8–12 small kernels
(plus 3 reductions for the alignment). At small batch sizes this launch overhead can be a
measurable fraction of step time — and it is *exactly* the regime the batch-size sweep
explores, so an unmeasured overhead will be misread as an optimizer property.

```python
import torch
def bench(opt, model, x, y, n=200):
    for _ in range(20):  # warmup
        opt.zero_grad(set_to_none=True); torch.nn.functional.cross_entropy(model(x), y).backward(); opt.step()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(n):
        opt.zero_grad(set_to_none=True); torch.nn.functional.cross_entropy(model(x), y).backward(); opt.step()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/n
```
Compare `MAL_SGD(adaptive=True)` against `torch.optim.SGD(..., foreach=True)` at bs 64 and
bs 2048. Report the ratio in the paper — reviewers ask for wall-clock, not just step counts.

**If overhead is material**, the optimization path is:
- batch the elementwise work with `torch._foreach_mul_` / `_foreach_add_` (the per-tensor
  reductions for the cosine cannot be fused as easily, but `torch._foreach_norm` covers the
  norms);
- consider CUDA graphs for the update (requires static shapes and no host syncs — §11.2 is
  a prerequisite);
- `torch.compile` on the *model* is orthogonal and safe; compiling the custom `step()` is
  not worth it until the above is exhausted.

Also worth a one-line experiment: `channels_last` memory format for the ResNet + AMP
typically gives a free speedup on Ampere+.

### 11.4 Multi-GPU correctness (needed before the LLM track)

Two non-obvious facts to preserve:

- **DDP is correct as-is and needs no extra communication.** DDP all-reduces gradients
  during `backward()`, so by the time `step()` runs, `p.grad` is already the globally
  averaged gradient. The alignment cosine is therefore computed on the true global gradient,
  and every rank derives the *same* β from the *same* inputs — the adaptive state stays in
  sync across ranks without any explicit synchronization. Do not add an all-reduce of β.
- **FSDP / ZeRO optimizer-state sharding will break.** All per-parameter state lives in
  `param_groups`, not `self.state` (§7.1), so sharding machinery will neither shard nor
  restore `momentum`/`beta`/`v`/`bc_prod`/`step`. Fixing §7.1 (move state into
  `self.state[p]`) is a **hard prerequisite** for FSDP, and for resuming a preemptible
  multi-node run. Under FSDP the gate would additionally operate on *parameter shards*
  rather than whole tensors, changing the alignment granularity — decide deliberately
  whether that is acceptable (it is closer to the per-output-unit variant of §8.5) and
  document it.

### 11.5 Scale-up experiments (the actual science on GPU)

1. **Batch-size × lr grid with the static-β ≈ 0.80 control** (§8.1–8.2). The
   noise-adaptivity hypothesis predicts MAL's edge **grows at bs 64 and shrinks at
   bs 2048**; Tier B (§9.3) already showed the ranking flipping at bs 512 on a small CNN.
   This is the single most decisive vision experiment left.
2. **β telemetry at scale** (§8.7): port `diagnostics/beta_diag.py` to
   `DEVICE="cuda"`, log per-step histograms to W&B (§10.2). Confirm at ResNet-50/CIFAR-100
   scale that (a) the mean still self-regulates near 0.80, (b) the β > 0.9 band is still
   used ~⅓ of the time, and (c) β responds to batch size as predicted. The
   per-layer β heatmap over a full 200-epoch cosine run is the paper's mechanism figure.
3. **CIFAR-100 / ResNet-50 confirmation** of the Tier-C conclusion (`MAL_ada` matches tuned
   SGD's peak and wins beyond it) at full 200-epoch scale with augmentation.
4. **Then** the transformer track: wire the AdamW classes (§8.4), validate on ViT/CIFAR
   with `betas=(0.9, 0.999)`, then LLM pretraining with `betas=(0.9, 0.95)`. For LLMs also
   handle: tiny 1-D params (`align_1d`), and **embedding tables with row-sparse gradients**
   — a whole-tensor cosine there is dominated by inactive rows, so per-row (per-output-unit)
   gating is likely necessary rather than optional.

### 11.6 Determinism and hygiene for publishable runs

`set_seed()` seeds python/numpy/torch/cuda, and DataLoader workers get
`set_worker_seed`, but cuDNN autotune is still nondeterministic. For the final
paper-table runs either accept it and report seed variance (3 seeds are already in the
sweep), or set `torch.use_deterministic_algorithms(True)` +
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and expect a slowdown. Do not mix deterministic and
nondeterministic runs within one reported table.
