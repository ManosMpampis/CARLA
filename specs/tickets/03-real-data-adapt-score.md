# 03: Real-data path — adapt stage + score handoff into untouched metrics

**What to build:** On real SMD telemetry: `adapt` briefly trains on machine windows, then `score` emits per-timestep anomaly-score arrays with window start/end indices into the EXISTING metrics stack unmodified. The scorer→evaluator handoff is the one new seam; everything downstream of it is frozen code.

**Blocked by:** 02 (synthetic pretrain tracer).

**Status:** ready-for-agent

- [ ] Adapt stage consumes machine-1-1 training windows via the new dataset class and completes a short schedule
- [ ] Score stage produces a per-timestep array whose length equals the test series length, plus start/end indices
- [ ] Overlap-aware aggregation verified: points covered by multiple sliding windows are counted exactly once per timestep
- [ ] Untouched metrics functions return the frozen metric-dictionary contract (all documented keys present)
- [ ] Thresholding uses train-side statistics only; no test labels reachable from the scoring path
