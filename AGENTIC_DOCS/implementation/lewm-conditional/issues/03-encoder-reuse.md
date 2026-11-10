# 03: Reuse trained encoders without changing their feature function

**What to build:** Reuse a trained masked-LeWM or AE encoder as a forecast backbone, and a trained LeWM encoder inside a simple AE, either as backbone plus identity encoder or as encoder with configurable freezing.

**Blocked by:** 01: Train and score the masked latent experiment correctly; 02: Compare future-query predictions against future latents.

**Status:** verified

- [x] Source architecture and feature metadata reconstruct or validate the exact pretrained feature function.
- [x] Frozen parameters and normalization buffers remain unchanged while downstream gradients flow.
- [x] AE encoder is trainable by default and supports explicit freezing; identity encoder preserves effective decoder geometry.
- [x] Saved task metadata rejects incompatible resume; resumed destination runs do not require the original source file.
- [x] Source selection supports matched runs and explicit overrides, with best-validation/best-training policy.
