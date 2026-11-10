# Conditional LeWM implementation

This folder contains local execution tickets, dependency tracking, coordination
notes and completion evidence for the approved conditional LeWM specification.
Keep implementation code in the existing source directories and permanent
assertion scripts in `tests/`.

Specification: [approved experiment](../../specs/lewm-conditional-experiments.md).
Baseline: `22dc54c1a8ac573c198e2517c46ba8915d33c9c7`.
Tracker: one Markdown file per ticket in `issues/`; statuses are
`ready-for-agent`, `claimed`, `implemented`, `verified`.

The user explicitly authorized ticket creation and execution together; the
approved spec supplies the acceptance criteria. Repository instructions place
agent artifacts under AGENTIC_DOCS, overriding the skill's default .scratch path.

| Ticket | Behavior | Blocked by | Implementation owner | Verification owner |
| --- | --- | --- | --- | --- |
| 01 | Correct masked latent LeWM training and scoring | None | model implementation agent | independent test agent |
| 02 | Guided future-query latent forecasting | 01 | model implementation agent | independent test agent |
| 03 | Exact encoder reuse in forecast and AE | 01, 02 | model implementation agent with runtime integration | independent test agent |
| 04 | Honest splits, short-series training and checkpoints | None | runtime implementation agent | independent test agent |
| 05 | Dataset selections, pooled evaluation and parallel jobs | 04 | runtime implementation agent | independent test agent |
| 06 | Reproducible full synthetic acceptance | 01–05 | architect coordinates fixes | independent test agent |
| 07 | Review, usage documentation and delivery | 06 | architect | separate standards/spec reviewers |

Agreed verification interfaces: model builder→forward/loss/score; dataset
selection→normalization/splits; normal training entry→validation/checkpoints/
reload/metrics; and suite entry→per-job selection/concurrency/resume. These are
the approved specification's acceptance interfaces. Tests use real modules and
generated format fixtures, not mocked implementation internals.

Ownership: model agent owns models, losses, reconstruction score primitives,
encoder-transfer helpers and training presets. Runtime agent owns train entry,
data, training/config/checkpoint/suite/scoring orchestration. Test agent owns
tests and the acceptance report. Architect owns this tracker/spec, integration,
review and final commits. Coordinate shared interfaces directly; agents do not
commit independently.

Implementation candidate: `6557f7a`. Independent acceptance passed all 29 groups
across four assertion scripts, including CUDA checks and real PSM/SMD samples;
TSB collections were verified with generated format/list fixtures only. See the
[acceptance report](../../reports/lewm_conditional_acceptance/REPORT.md).
All seven tickets are verified. Separate standards and specification reviewers
found no actionable findings. See [review.md](review.md) for the reviewed range
and final reports. The implementation and acceptance evidence are committed
on the existing branch.
