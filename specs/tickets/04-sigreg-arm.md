# 04: SIGReg anti-collapse arm

**What to build:** `anti_collapse: sigreg` becomes selectable by config: sliced-projection Gaussianity regularization stabilizes latent training with λ as the single effective hyperparameter. SIGReg is a training-time regularizer only — the anomaly-score path never routes through it.

**Blocked by:** 02 (synthetic pretrain tracer).

**Status:** ready-for-agent

- [ ] Config selecting sigreg trains the synthetic tracer end-to-end
- [ ] Latent variance diagnostic logged and stays above a collapse floor for the short schedule (compare: same run without regularization collapses toward constant latents)
- [ ] λ exposed and honored in config
- [ ] Scoring works identically with the regularizer absent at inference time
