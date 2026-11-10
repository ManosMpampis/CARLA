# 05: Run dataset selections with pooled evaluation and bounded concurrency

**What to build:** Use one trainer for PSM, SMD single/all/all-one and TSB-U/M single/CSV-list jobs, with preflight, isolated parallel workers, failure reporting and resumable suite manifests.

**Blocked by:** 04: Train with honest validation and the requested checkpoint selections.

**Status:** verified

- [x] PSM needs no filename; SMD all fans out and all-one trains one boundary-safe corpus.
- [x] Pooled SMD uses per-machine train normalization, calibration, evaluation and macro summaries with one model.
- [x] TSB collections have separate identities/output namespaces; file_name manifests preserve selection and evaluate only the test suffix.
- [x] Every independent job gets fresh model/optimizer state; configurable concurrency defaults to one.
- [x] Missing/duplicate selections fail preflight; runtime failures are recorded and remaining jobs continue; completed jobs are skipped on resume.

Implementation: `utils/jobs.py`, `utils/run_evaluation.py`, `train.py`,
`data/jepa_dataset.py`, and native compatibility adapters. The independent suite
checks pass selection/preflight, bounded parallel execution with exact isolated
weights, runtime failure continuation/retry, completed-job skipping, pooled
calibration/macro results, TSB U/M test suffixes, and CLI train-to-score/artifact
regeneration. The final acceptance run reports all 29 assertion groups passing;
see the test owner's `suite_checks.json` evidence under the acceptance report.
Usage: [runtime](../runtime.md).
