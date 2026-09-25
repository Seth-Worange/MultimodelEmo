# Q2 Autonomous R&D Final Report

This report is a source freeze of the completed local integration stage. The supplied data is now
at `code/data`. The selected Windows environment exists at `D:\\condaData\\envs_dirs\\my_env01`,
but it currently contains a CPU-only PyTorch build, so GPU training metrics remain pending.

## Implemented

- isolated branch `shuke/q2-fuse-missing` from local `main`;
- selective FUSE, data, mask, train, evaluate, robustness, and inference integration;
- train-only audio/vision normalization with checkpoint persistence;
- frozen B0/B1/B2 configs, protocol, audit, registry, and focused normalization test.
- data root corrected to `code/data`; the CPU environment can run without installing
  `transformers` when precomputed teacher features are present.

## CPU smoke evidence

One epoch completed for B0, B1, and B2. B1 evaluation and a reduced validation robustness
scan also completed. These runs validate the execution path only and are not final model
comparisons; they used CPU and one epoch.

## Pending on the data machine

Run B0, B1, and B2 with seed 2026, evaluate the validation robustness suite, inspect Text40,
neutral-class F1 and regression magnitude bins, then repeat only the selected candidate with seeds
2027 and 2028. Populate `final_summary.csv`, `final_multiseed_summary.csv`, plots, and this report
with measured values. The test split is evaluated once after selection.

## Integrity statement

No test-set selection, fabricated metric, push, merge, force push, or remote PR was performed.
