# 12: Cutover — deletion manifest execution

**What to build:** Once everything above runs end-to-end: remove the legacy contrastive training loop, unused losses, old entry scripts/drivers, superseded config trees, and clustering-selection code. Preserve the four contrastive dataset classes intact-but-unused per owner decision, plus Logger, GradientMonitor, checkpoint utilities, metrics package, loaders, and SubAnomaly machinery. Refresh docs to the new reality.

**Blocked by:** 11 (reporting + reference numbers).

**Status:** ready-for-agent

- [ ] Manifest deletions applied; nothing remaining imports deleted symbols
- [ ] Contrastive dataset classes remain present, unused, importable
- [ ] Every published arm config still completes its smoke run after deletion
- [ ] AGENTS.md / design docs updated: legacy gotchas that die with the deleted code are removed or rewritten
