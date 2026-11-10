# Conditional LeWM independent acceptance

This folder holds independent verification results for the approved conditional
LeWM experiment: commands, architecture evidence, objective checks, fixture
training/validation/test results, transfer provenance and remaining limitations.
Permanent assertion scripts belong in `tests/`; generated checkpoints, dataset
fixtures and large experiment outputs remain under `/tmp`.

Status: **passed, 2026-11-10**. See [REPORT.md](REPORT.md) for the independent
29-group acceptance, commands, architecture shapes, training/validation/test
results, transfer provenance and limitations. The four authoritative JSON
reports and their logs are retained alongside the report.

`initial_models.json` is an archived-baseline replay: the first execution was
already green after concurrent implementation fixes; the same eight initial
groups were subsequently verified to fail against the untouched baseline.
`write_report.py` regenerates the report and loss figure from final JSON.
