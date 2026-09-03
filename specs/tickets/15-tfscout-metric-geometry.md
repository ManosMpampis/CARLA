# 15: Metric geometry default vs mixture variant

**What to build:** Per-machine normalized-center metric learning with variance guard plus inverse-distance push on synthetic injections as default, compared against a small multi-component mixture-energy variant, so multi-regime machines are modeled without cross-machine leakage.

**Blocked by:** 13 (trunk plus open-loop baseline).

**Status:** ready-for-agent

- [ ] Default center initializes from training embeddings, stays fixed, and scores distance-to-identity per position
- [ ] Synthetic push term separates injected probes from clean windows without test labels
- [ ] Mixture variant with few components trains per machine as a separate arm
- [ ] Tournament promotes one geometry on honest metrics with acceptable per-machine variance
