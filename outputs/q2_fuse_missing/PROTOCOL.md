# Frozen experiment protocol

All candidates use seed 2026 for discovery, AdamW, batch size 32, at most 50 epochs, patience
8, learning rate `1e-3`, the same FUSE architecture and auxiliary weights, and validation-only
checkpoint selection over clean/local/whole/interval views. The test split is not used for
selection. The three candidates are:

| candidate | normalization | structured missing |
|---|---:|---:|
| B0 | off | off |
| B1 | train-only audio/vision stats | off |
| B2 | train-only audio/vision stats | local + interval dominant, low-rate whole |

`config/q2_fuse_b0.yaml`, `config/q2_fuse_b1.yaml`, and `config/q2_fuse_b2.yaml` freeze these
choices. B2 keeps text in the training missing distribution through the model's text mask path;
the interval plan is explicitly audio/vision because the supplied aligned special set reports
no text missing. The validation robustness suite remains analysis-only: clean, audio+vision
40% middle, text 40% middle, vision 40% middle, text+vision 40% middle, and audio+vision 60%
middle. Each row records Accuracy, macro-F1, MAE, Pearson, and per-class F1.

Acceptance uses paired validation results. A candidate must improve Robust-F1 materially while
keeping clean macro-F1 loss within about 0.005 and clean MAE degradation within about 0.01,
unless the robustness gain clearly justifies the trade-off. These are engineering screening
rules, not claims of statistical significance.
