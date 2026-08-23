# 10: SubAnomaly Calibrator — train-only thresholds + calibrated fusion

**What to build:** Injected synthetic anomalies (`SubAnomaly`) act purely as held-out probes: the Calibrator consumes clean-train and injected-probe score distributions to produce detection thresholds (train quantiles) and calibrated level/signal fusion weights replacing the plain mean. No test labels are reachable from this path — that guarantee is the ticket's core.

**Blocked by:** 03 (real-data path).

**Status:** ready-for-agent

- [ ] Probe views (clean-train + injected-anomaly) scored through the normal scoring path
- [ ] Thresholds derived exclusively from training-side distributions and persisted alongside checkpoints
- [ ] Fusion weights calibrated on probe separation; plain mean remains the fallback when calibration data is thin
- [ ] Verified: calibrator inputs trace only to train-side artifacts; scores on injected probes separate from clean-train scores on machine-1-1
