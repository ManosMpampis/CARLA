# 06: Soft-codebook arm + prototype anomaly signals

**What to build:** `anti_collapse: codebook` becomes selectable: a learned prototype dictionary with soft-attention routing and k-means warmup initialization stabilizes training (SC-JEPA-style) and contributes two extra anomaly signals — distance-to-nearest-prototype and attention entropy — into the scorer's fusion.

**Blocked by:** 02 (synthetic pretrain tracer), 03 (real-data path).

**Status:** ready-for-agent

- [ ] Config selecting codebook trains the synthetic tracer end-to-end
- [ ] Prototypes initialized by k-means warmup on first-epoch latents; warmup phase observable in logs
- [ ] Prototype-distance and attention-entropy signals computed per token at scoring time
- [ ] Score fusion remains mean across signals/levels until the Calibrator ticket lands
