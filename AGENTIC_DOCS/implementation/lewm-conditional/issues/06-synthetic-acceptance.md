# 06: Verify the complete experiment with independent synthetic acceptance

**What to build:** Independently exercise the normal experiment entries for all four frameworks, every transfer route and the dataset/job matrix, recording architecture, objective, validation and calibrated/oracle test results.

**Blocked by:** 01, 02, 03, 04, 05.

**Status:** verified

- [x] Plain Python assertion scripts exercise real public interfaces; archived-baseline replay records initial regression failures without modifying the baseline.
- [x] Short training, fixed-batch optimization, score reload, resume, checkpoint selections and coverage pass.
- [x] PSM/SMD/TSB-U/TSB-M format fixtures and sequential/parallel suites pass; real PSM/SMD samples pass. Real TSB collections are not installed and are explicitly recorded as a limitation.
- [x] Report records commands, component losses, architecture shapes, metrics, artifacts and material limitations.

Independent verification on 2026-11-10: all 29 groups passed across
`tests/verify_conditional_models.py`, `tests/verify_conditional_runtime.py`,
`tests/verify_conditional_transfer.py` and `tests/verify_conditional_suites.py`.
Both module BF16 and the normal Trainer/GradScaler path passed on CUDA.
See [acceptance report](../../../reports/lewm_conditional_acceptance/REPORT.md)
and its authoritative JSON/log evidence. No implementation source was edited
by the acceptance owner, and no independent commits were made.
