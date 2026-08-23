# 02: Thin tracer — synthetic end-to-end pretrain

**What to build:** Running one tiny synthetic-data YAML through the `pretext` stage trains a minimal self-JEPA (dense L1 against stop-gradient targets) end-to-end on CPU in minutes: config → factories → dataset → Trainer → checkpoints → TensorBoard. This is the vertical tracer bullet every later ticket extends.

**Blocked by:** 01 (registry prefactor).

**Status:** ready-for-agent

- [ ] Synthetic tiny config runs the full pretext stage deterministically on CPU within minutes
- [ ] Training loss decreases measurably across the short schedule
- [ ] Checkpoint written in the existing resume format; killing and rerunning resumes without a loss discontinuity
- [ ] TensorBoard events include loss scalars and the model graph
- [ ] Mixed-precision flag accepted by the Trainer (off by default; fp32 path remains the deterministic default)
