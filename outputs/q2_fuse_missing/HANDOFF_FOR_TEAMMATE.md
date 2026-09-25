# Q2 Autonomous R&D handoff

## Current source state

- Branch: `shuke/q2-fuse-missing`
- Latest source commit before this handoff: `f661317`
- Earlier integration commit: `bffa816`
- No push, merge, or PR was performed.
- Data root: `D:\\PycharmProjects\\26MATH\\code\\data`
- Python: `D:\\condaData\\envs_dirs\\my_env01\\python.exe`
- Current PyTorch is CPU-only and is the intended execution environment.

## Completed work

1. Selective FUSE integration from the teammate branch.
2. Missing-aware HMF/MRC/MDF path with cross reconstruction.
3. Train-only normalization implementation and checkpoint persistence.
4. B0/B1/B2 discovery with seed 2026.
5. Full 50-case validation robustness scans for B0/B1/B2.
6. LR screening for `3e-4`, `5e-4`, `1e-3`, `2e-3`.
7. Formal final seed 2027 training completed.
8. Data, augmentation, network, FUSE demos and compile checks passed.

## Measured discovery results

| candidate | best epoch | selection score | clean macro-F1 | clean MAE | fixed R1-R5 Robust-F1 | Robust-MAE |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 2 | 0.2941 | 0.6303 | 0.6112 | 0.6032 | 0.6155 |
| B1 | 2 | 0.3124 | 0.6052 | 0.6465 | 0.5690 | 0.6731 |
| B2 | 3 | 0.3109 | 0.6035 | 0.6362 | 0.5714 | 0.6549 |

LR screening selection scores:

- `3e-4`: 0.2960
- `5e-4`: 0.2954
- `1e-3`: 0.2941 (selected)
- `2e-3`: 0.3045

The selected final protocol is B0 FUSE, normalization off, structured missing off, LR `1e-3`.
B1 and B2 are negative results and must remain in the final report.

Formal seed 2027 is present at `outputs/q2_fuse_missing/Final/s2027/`:
best epoch 3, selection score 0.3028, clean macro-F1 0.6156, clean MAE 0.6366.

## Remaining execution

Run seed 2028 with the frozen protocol:

```powershell
& 'D:\condaData\envs_dirs\my_env01\python.exe' -m scripts.train --config config\q2_fuse_b0.yaml --device cpu --seed 2028 --lr 0.001 --output-dir outputs\q2_fuse_missing\Final\s2028
```

Then run the full robustness suite for the selected final checkpoints:

```powershell
& 'D:\condaData\envs_dirs\my_env01\python.exe' -m scripts.robustness --checkpoint outputs\q2_fuse_missing\B0\s2026\best.pt --data-root ..\data --device cpu --batch-size 128 --output outputs\q2_fuse_missing\Final\s2026_robustness.csv
& 'D:\condaData\envs_dirs\my_env01\python.exe' -m scripts.robustness --checkpoint outputs\q2_fuse_missing\Final\s2027\best.pt --data-root ..\data --device cpu --batch-size 128 --output outputs\q2_fuse_missing\Final\s2027_robustness.csv
& 'D:\condaData\envs_dirs\my_env01\python.exe' -m scripts.robustness --checkpoint outputs\q2_fuse_missing\Final\s2028\best.pt --data-root ..\data --device cpu --batch-size 128 --output outputs\q2_fuse_missing\Final\s2028_robustness.csv
```

Use validation only for all selection and robustness calculations. Do not run test evaluation
until the final model is frozen. Compute mean and standard deviation across seeds 2026/2027/2028
for clean Accuracy, Macro-F1, MAE, Pearson, Robust-F1, Robust-MAE, and the fixed R1-R5 rows.

## Required final artifacts

- `outputs/q2_fuse_missing/final_summary.csv`
- `outputs/q2_fuse_missing/final_multiseed_summary.csv`
- `outputs/q2_fuse_missing/FINAL_REPORT.md`
- confusion matrices and neutral/strong-sentiment regression diagnostics
- final tests, compile check, `git diff --check`
- one local source-freeze commit

The final report must state the B1/B2 negative results, the LR screening table, seed mean ± std,
comparison against the historical FUSE reference, and the final limitations. It must state
`push: NO`, `merge: NO`, `PR: NO`.
