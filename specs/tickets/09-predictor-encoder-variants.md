# 09: Variant arms — GRU predictor + transformer encoder

**What to build:** Comparison arms become config-selectable: a GRU predictor alternative to the TCN, and the transformer token encoder as a capacity comparison (kept per owner decision, never the default). Both train under SIGReg so neither runs unstabilized.

**Blocked by:** 04 (SIGReg arm).

**Status:** ready-for-agent

- [ ] `predictor: tcn|gru` selectable by config; both variants complete the synthetic tracer
- [ ] Transformer-encoder arm config published and completes the synthetic tracer
- [ ] Convolutional pyramid + TCN remains the default path in all shared configs
