# 08: Two-stage learning part B — adaptation modes

**What to build:** Stage-B adaptation consumes a stage-A pretrained checkpoint with two selectable modes: `frozen` (encoder frozen, predictor + scoring path train) and `finetune` (everything trains). Default frozen per the agreed spec. The official headline protocol stays per-machine.

**Blocked by:** 07 (stage-A masked pretraining).

**Status:** ready-for-agent

- [ ] Adaptation loads a stage-A checkpoint and continues training from it (resume-safe)
- [ ] Frozen mode: encoder parameters bit-identical before/after a short adaptation run
- [ ] Finetune mode: all parameters update
- [ ] Mode selectable purely by config; per-machine protocol remains the default path
