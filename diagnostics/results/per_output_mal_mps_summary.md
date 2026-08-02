# Per-output MAL validation summary

- Runs: 144 (72 exact scope pairs)
- Pairing/data-integrity issues: 0
- Deltas are `per-output - per-tensor` in percentage points.
- Cell entries are paired means; condition-level 95% intervals cluster the hyperparameter grid within seed.

| condition | c | lr | best-val delta | selected-test delta | output wins |
|---|---:|---:|---:|---:|---:|
| clean | 0.1 | 0.05 | -1.07 | -0.35 | 0/3 |
| clean | 0.1 | 0.2 | +0.83 | +1.32 | 2/3 |
| clean | 0.1 | 0.6 | +0.47 | +0.52 | 2/3 |
| clean | 0.3 | 0.05 | -0.07 | -0.57 | 1/3 |
| clean | 0.3 | 0.2 | -5.03 | -3.85 | 1/3 |
| clean | 0.3 | 0.6 | +1.57 | +0.97 | 2/3 |
| clean | 1.0 | 0.05 | -1.03 | -0.03 | 1/3 |
| clean | 1.0 | 0.2 | +1.83 | +2.18 | 2/3 |
| clean | 1.0 | 0.6 | -0.40 | -0.25 | 1/3 |
| noise20 | 0.1 | 0.05 | +0.00 | +0.15 | 3/5 |
| noise20 | 0.1 | 0.2 | +0.82 | +0.66 | 3/5 |
| noise20 | 0.1 | 0.6 | +1.84 | +0.05 | 5/5 |
| noise20 | 0.3 | 0.05 | -0.16 | +0.29 | 1/5 |
| noise20 | 0.3 | 0.2 | +1.04 | +2.18 | 3/5 |
| noise20 | 0.3 | 0.6 | +1.02 | +0.16 | 4/5 |
| noise20 | 1.0 | 0.05 | -0.10 | +1.06 | 2/5 |
| noise20 | 1.0 | 0.2 | +2.62 | +3.11 | 4/5 |
| noise20 | 1.0 | 0.6 | +1.72 | +1.31 | 4/5 |

## Grid-averaged result by condition

| condition | seeds | best-val delta (95% CI) | selected-test delta (95% CI) | output wins |
|---|---:|---:|---:|---:|
| clean | 3 | -0.32 [-2.82, +2.17] | -0.01 [-2.61, +2.59] | 12/27 (0 ties) |
| noise20 | 5 | +0.98 [+0.31, +1.65] | +1.00 [-0.13, +2.12] | 29/45 (2 ties) |

Optimizer-only output/tensor runtime ratio across suites: 0.991 (range 0.955–1.032).
