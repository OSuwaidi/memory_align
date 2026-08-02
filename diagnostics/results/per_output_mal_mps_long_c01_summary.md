# Per-output MAL validation summary

- Runs: 12 (6 exact scope pairs)
- Pairing/data-integrity issues: 0
- Deltas are `per-output - per-tensor` in percentage points.
- Cell entries are paired means; condition-level 95% intervals cluster the hyperparameter grid within seed.

| condition | c | lr | best-val delta | selected-test delta | output wins |
|---|---:|---:|---:|---:|---:|
| clean | 0.1 | 0.2 | -0.62 | +0.32 | 1/3 |
| noise20 | 0.1 | 0.2 | -1.00 | -0.24 | 1/3 |

## Grid-averaged result by condition

| condition | seeds | best-val delta (95% CI) | selected-test delta (95% CI) | output wins |
|---|---:|---:|---:|---:|
| clean | 3 | -0.62 [-3.82, +2.59] | +0.32 [-3.76, +4.40] | 1/3 (1 ties) |
| noise20 | 3 | -1.00 [-8.94, +6.94] | -0.24 [-6.31, +5.82] | 1/3 (0 ties) |

Optimizer-only output/tensor runtime ratio across suites: 0.969 (range 0.969–0.969).
