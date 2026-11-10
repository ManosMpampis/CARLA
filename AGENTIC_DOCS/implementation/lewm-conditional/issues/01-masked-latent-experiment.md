# 01: Train and score the masked latent experiment correctly

**What to build:** Run the approved corruption-to-clean LeWM experiment with auxiliary-predicted actions, a correctly packed one-STFT pyramid, unrestricted latents, selectable KL/L1/MSE/L2 prediction loss and independently selected scores, defaulting to KL/L1.

**Blocked by:** None (can start immediately).

**Status:** implemented

- [x] The default run is latent-only, with predicted masks in training and scoring, supervised by BCE on synthetic masks.
- [x] Spectral packing roundtrips without mixing complex, time, batch or feature coordinates.
- [x] Exact latent shapes, configurable objective terms, finite gradients including zero-error L2, and independent scoring are verified.
- [x] Multiple legal temporal lengths work; SIGReg receives unrestricted latent outputs; shared target path remains trainable.

Implementation evidence: `tests/verify_conditional_models.py` passes the
independent model-interface assertions, literal objective/scoring equations,
complex spectral-coordinate roundtrips, mask independence, target gradients,
unrestricted outputs, optional raw objective, and fixed-batch optimization.
The test agent additionally verified CUDA BF16 transforms and FP32 scoring.
