# Q2 Autonomous R&D Final Report

This report is a source freeze of the completed local integration stage. Training and metric
claims are intentionally pending because the supplied data directory and the documented PyTorch
CUDA environment are not present in this workspace.

## Implemented

- isolated branch `shuke/q2-fuse-missing` from local `main`;
- selective FUSE, data, mask, train, evaluate, robustness, and inference integration;
- train-only audio/vision normalization with checkpoint persistence;
- frozen B0/B1/B2 configs, protocol, audit, registry, and focused normalization test.

## Pending on the data machine

Run B0, B1, and B2 with seed 2026, evaluate the validation robustness suite, inspect Text40,
neutral-class F1 and regression magnitude bins, then repeat only the selected candidate with seeds
2027 and 2028. Populate `final_summary.csv`, `final_multiseed_summary.csv`, plots, and this report
with measured values. The test split is evaluated once after selection.

## Integrity statement

No test-set selection, fabricated metric, push, merge, force push, or remote PR was performed.
