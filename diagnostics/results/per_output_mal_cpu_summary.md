# Per-output MAL validation summary

- Runs: 48 (24 exact scope pairs)
- Pairing/data-integrity issues: 0
- Deltas are `per-output - per-tensor` in percentage points.
- 95% intervals are paired t intervals and are descriptive with only three seeds.

| condition | c | lr | best-val delta | selected-test delta | output wins |
|---|---:|---:|---:|---:|---:|
| clean | 0.3 | 0.05 | -0.07 | -0.57 | 1/3 |
| clean | 0.3 | 0.2 | -5.03 | -3.87 | 1/3 |
| clean | 0.3 | 0.6 | +1.20 | +0.57 | 2/3 |
| clean | 1.0 | 0.05 | -1.03 | -0.03 | 1/3 |
| clean | 1.0 | 0.2 | +1.83 | +2.18 | 2/3 |
| clean | 1.0 | 0.6 | -0.30 | -0.43 | 1/3 |
| noise20 | 0.3 | 0.2 | +1.83 | +3.00 | 2/3 |
| noise20 | 1.0 | 0.2 | +0.87 | +0.32 | 3/3 |

Optimizer-only output/tensor runtime ratio across suites: 0.955 (range 0.951–0.957).
