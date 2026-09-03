# 18: Full tournament plus reporting

**What to build:** Complete head tournament across all tactics with per-machine normal-only training, train-only thresholds, honest vs point-adjust separation, and mandatory no-training baselines, so the promoted tactic set is defensible.

**Blocked by:** 14 (KL channel decision), 15 (metric geometry), 16 (shape-loss ablation), 17 (closed-loop action variant).

**Status:** ready-for-agent

- [ ] Every tactic runs on the fixed trunk with one swap at a time per machine
- [ ] Thresholds derive from clean-train plus injected-probe distributions only
- [ ] Headline honest metrics and separated comparability metrics both reported with baselines
- [ ] Promotion log records which tactics beat no-training baseline with non-trivial fusion weight
