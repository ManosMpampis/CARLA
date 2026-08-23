# 11: Honest-vs-point-adjust reporting + published reference numbers

**What to build:** Final reporting shape: window-level AUROC/AP and point-level F1 without point-adjust as the honest headline; all point-adjust metrics clearly separated for literature comparability only. Plus a reference-numbers document harvesting published SMD/PSM results (canonical methods plus recent SOTA) with provenance flags distinguishing paper-reported from benchmark-reproduced conventions.

**Blocked by:** 10 (SubAnomaly Calibrator).

**Status:** ready-for-agent

- [ ] Evaluation output groups metrics into honest headline vs point-adjust-comparability sections
- [ ] No tuning path reads test-set statistics; documented and greppable
- [ ] Reference-numbers document covers CARLA, USAD, OmniAnomaly, MTAD-GAT, GDN, Anomaly Transformer plus recent-SOTA entries, each flagged with provenance and metric convention
