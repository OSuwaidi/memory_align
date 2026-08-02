# Per-output MAL validation summary

- Runs: 12 (6 exact scope pairs)
- Pairing/data-integrity issues: 0
- Deltas are `per-output - per-tensor` in percentage points.
- Cell entries are paired means; condition-level 95% intervals cluster the hyperparameter grid within seed.

| condition | c | lr | best-val delta | selected-test delta | output wins |
|---|---:|---:|---:|---:|---:|
| clean | 1.0 | 0.2 | +1.62 | +0.94 | 3/3 |
| noise20 | 1.0 | 0.2 | -1.62 | -0.95 | 0/3 |

## Grid-averaged result by condition

| condition | seeds | best-val delta (95% CI) | selected-test delta (95% CI) | output wins |
|---|---:|---:|---:|---:|
| clean | 3 | +1.62 [-3.53, +6.76] | +0.94 [-4.71, +6.60] | 3/3 (0 ties) |
| noise20 | 3 | -1.62 [-4.35, +1.12] | -0.95 [-3.19, +1.29] | 0/3 (0 ties) |

Optimizer-only output/tensor runtime ratio across suites: 0.961 (range 0.961–0.961).
