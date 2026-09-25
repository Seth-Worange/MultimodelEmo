# FUSE + missing-modality integration audit

## Scope and provenance

The branch starts at local `main` (`f88c813`). FUSE and the validated masking/data
helpers were copied selectively from `shuke/q2-cmad`; the CMAD distillation files were
not merged. The audit was performed before adding the B0/B1/B2 protocol files.

## Data and shapes

| item | contract |
|---|---|
| token input | `(B, 3, 50)` from `text_bert`; model receives the token id plane `(B, 50)` |
| text teacher | `(B, 50, 768)` when `text_mode=bert` |
| audio | `(B, 50, 74)` |
| vision | `(B, 50, 35)` |
| masks | boolean `(B, 50)` for text/audio/vision |
| labels | class ids `0/1/2`; signed regression target |
| FUSE output | three logits, sentiment, magnitude, modality weights, valid steps, factor/reconstruction diagnostics |

## FUSE path

- T/A/V inputs are encoded by independent GRU paths to width 128.
- HMF creates shared/private/noise projections per modality.
- MRC applies a variational bottleneck and reconstruction per available position.
- MDF combines sample modulation, factor coefficients, branch attention, and noise gating.
- Cross reconstruction uses the available fused representation to reconstruct a modality that
  is present in the clean pair but missing in the masked pair.

## Mask semantics

`True` means the modality is available at that word position. `mask_batch` clones all input
fields before changing masks and zeroing the corresponding values. Padding positions remain
invalid. FUSE gates factorization, reconstruction, cross reconstruction, and fusion with these
masks; the temporal pooling mask is the union of available modalities.

## Normalization

`utils.normalization` fits per-feature audio/vision mean and standard deviation on valid
training positions only. It applies the same statistics to validation/test data and restores
missing/padding positions to exact zero after normalization. B0 leaves normalization off;
B1/B2 enable it. Statistics are stored in `best.pt` and loaded by evaluate, robustness, and
specialized inference, preventing train/validation leakage and checkpoint mismatch.

## Known execution blockers

The checked out workspace has no `data/` directory. The documented Windows environment
`C:\\Anaconda3\\envs\\pytorch` is absent, and WSL Python 3.14 has no PyTorch, NumPy, pip, or
pytest. GitHub SSH and HTTPS connections also did not complete in this session. Therefore
training metrics must be produced on the machine containing the supplied data and CUDA
environment; no metric is fabricated here.

## Follow-up checks

Before training, run the pure Python demos in `utils/augmentation.py` and `model/fuse_net.py`,
then the repository tests in the documented PyTorch environment. Confirm that each run records
`run.json`, `metrics.csv`, and `best.pt`, and run validation robustness only on the validation
split.
