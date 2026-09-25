# Autonomous R&D log

| iteration | problem/evidence | hypothesis | single modification | result | decision |
|---|---|---|---|---|---|
| 0 | local `main` was an early scaffold; FUSE/missing code existed only on the teammate branch | selectively port the FUSE and missing path, preserving branch isolation | create `shuke/q2-fuse-missing`; copy FUSE/data/masking/train/eval files | source integration available for audit | keep |
| 1 | audio/vision scales differ by orders of magnitude; missing values are represented by zero plus a mask | train-only per-feature normalization should improve conditioning if zero semantics are restored after normalization | add `utils/normalization.py`, wire stats through train/checkpoint/eval/robustness/infer | code path and focused unit test added; full run awaits data/PyTorch environment | keep pending experiment |
| 2 | B0/B1/B2 need comparable, explicit settings | separate config files prevent protocol drift | add frozen B0/B1/B2 YAML configs and PROTOCOL.md | protocol is reviewable and reproducible | keep pending metrics |
