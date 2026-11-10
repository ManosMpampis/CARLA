# 07: Review and deliver the conditional experiment workflow

**What to build:** Deliver a reviewed, documented and committed implementation of the approved specification with runnable usage examples and verified ticket status.

**Blocked by:** 06: Verify the complete experiment with independent synthetic acceptance.

**Status:** verified

- [x] Separate standards and spec reviewers inspect changes from the pinned baseline; material findings are resolved and relevant checks rerun.
- [x] Usage documents the conditional information flow, independent losses/scores, transfer, dataset selection, checkpoint policy and concurrency.
- [x] Final source/format checks pass; source paths and frozen metrics remain intact.
- [x] Work is committed on the current branch and final delivery links spec, tickets and acceptance evidence.

Evidence: implementation candidate `6557f7a`, independent 29-group acceptance
report, separate standards/spec review with zero actionable findings, syntax
compilation and `git diff --check`. Final documentation closeout is committed
separately from the implementation candidate.
