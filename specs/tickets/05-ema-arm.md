# 05: EMA teacher arm

**What to build:** `anti_collapse: ema` becomes selectable: an exponential-moving-average target encoder provides stop-gradient targets with layer-normed target features and a constant momentum default, matching shipped V-JEPA practice. Validation statistics computed in eval mode (guards the BatchNorm-inflated-validation failure documented in the LeWorldModel reproduction).

**Blocked by:** 02 (synthetic pretrain tracer).

**Status:** ready-for-agent

- [ ] Config selecting ema trains the synthetic tracer end-to-end
- [ ] Target-encoder weights provably trail online weights (EMA update observable across steps)
- [ ] Targets layer-normed before loss; stop-gradient verified (no gradients reach target branch)
- [ ] Momentum value configurable; validation loss logged from eval-mode forward passes
