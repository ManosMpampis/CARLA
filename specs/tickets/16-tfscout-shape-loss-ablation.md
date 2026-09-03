# 16: Shape-loss ablation on proposed parts

**What to build:** Part-restricted reconstruction comparing pointwise error against non-negative shape divergence with temporal-distortion ablation, so spikes and morphology shifts are both caught without full-window cost.

**Blocked by:** 13 (trunk plus open-loop baseline).

**Status:** ready-for-agent

- [ ] Reconstruction loss applies only inside proposed part masks from interval detections
- [ ] Shape-divergence weight and smoothing settings exposed and honored
- [ ] Temporal-distortion term runs as a separate ablation arm, not the default
- [ ] Ablation promotes one weighting on honest metrics without harming spike detection
