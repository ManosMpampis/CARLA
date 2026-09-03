# 13: Trunk plus open-loop H1/H4 baseline end-to-end

**What to build:** Fixed trunk (TF-grid representation, dual pyramids with top-down fusion, every-level modulation plus bottleneck attention, three-axis part-vs-whole mismatch maps) with training-only reconstruction-focus and box-focus heads, detection and metric-learning primary heads, dense canonical scores and honest metrics on one machine.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Trunk builds from config with one anti-collapse mechanism per run and disposable projector excluded from scoring
- [ ] Each auxiliary head and each primary head trains and scores alone via enable flags
- [ ] Dense per-timestep scores aggregate cover-count-aware and satisfy the frozen metrics contract
- [ ] Single-machine smoke run yields honest metrics plus no-training baseline
