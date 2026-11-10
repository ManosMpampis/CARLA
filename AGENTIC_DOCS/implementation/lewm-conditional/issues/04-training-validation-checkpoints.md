# 04: Train with honest validation and the requested checkpoint selections

**What to build:** Run training with optimization-train-only normalization, reproducible validation, explicit short-series fallback and separate last, primary-best and per-aggregation oracle-F1 checkpoints.

**Blocked by:** None (can start immediately).

**Status:** verified

- [x] Synthetic and real-format train/validation/test fixtures reuse the training-fitted normalization without test leakage.
- [x] Disjoint validation is used when available; otherwise calibration uses all training rows and primary selection uses best training loss.
- [x] Sample-weighted component/total summaries and frozen validation buffers are verified.
- [x] Last, best-validation or best-training, and best-oracle-F1 per evaluated aggregation remain separately identifiable and reloadable.
- [x] Validation RNG is isolated; resume restores the run and semantic metadata is validated.

Implementation: `data/jepa_dataset.py`, `train.py`, `utils/trainer.py`,
`utils/checkpointing.py`, `utils/randomness.py`, `utils/run_evaluation.py`.
`tests/verify_conditional_runtime.py` passed all seven slices, including two-epoch
AE/VAE/masked/guided training and all aggregation strategies. Independent final
verification is tracked by ticket 06. Usage: [runtime](../runtime.md).
