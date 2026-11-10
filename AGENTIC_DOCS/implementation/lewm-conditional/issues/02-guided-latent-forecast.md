# 02: Compare future-query predictions against future latents

**What to build:** Run guided LeWM with future-derived queries, paired historical keys/values, approximately 2:1 stride-aligned crops and a small transpose module predicting the exact future latent shape.

**Blocked by:** 01: Train and score the masked latent experiment correctly.

**Status:** verified

- [x] Unequal history/future token lengths work without pairing future keys with historical values.
- [x] Default predictor contains a length-preserving transpose convolution and no default raw decoder.
- [x] Default stride is 1; coarser latent score maps interpolate to the correct raw target interval.
- [x] Guide/history perturbations exercise the intended conditioning paths; score maps are FP32.

Implementation evidence: `tests/verify_conditional_models.py` checks actual
attention inputs, variable-length 2:1 crops, transpose geometry, scalar-grid
interpolation, fixed-batch objective reduction and guide/history perturbations.
Historical K/V share their original positions; future Q uses relative position
at the forecast gap, with a shared origin at the history endpoint.
