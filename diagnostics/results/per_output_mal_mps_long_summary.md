# Per-output MAL validation summary

- Runs: 24 (12 exact scope pairs)
- Pairing/data-integrity issues: 0
- Deltas are `per-output - per-tensor` in percentage points.
- Cell entries are paired means; condition-level 95% intervals cluster the hyperparameter grid within seed.

| condition | c | lr | best-val delta | selected-test delta | output wins |
|---|---:|---:|---:|---:|---:|
| clean | 0.1 | 0.2 | -0.62 | +0.32 | 1/3 |
| clean | 1.0 | 0.2 | +1.62 | +0.94 | 3/3 |
| noise20 | 0.1 | 0.2 | -1.00 | -0.24 | 1/3 |
| noise20 | 1.0 | 0.2 | -1.62 | -0.95 | 0/3 |

## Grid-averaged result by condition

| condition | seeds | best-val delta (95% CI) | selected-test delta (95% CI) | output wins |
|---|---:|---:|---:|---:|
| clean | 3 | +0.50 [-3.16, +4.16] | +0.63 [-3.59, +4.85] | 4/6 (1 ties) |
| noise20 | 3 | -1.31 [-5.70, +3.09] | -0.60 [-2.98, +1.79] | 1/6 (0 ties) |

Optimizer-only output/tensor runtime ratio across suites: 0.965 (range 0.961–0.969).
