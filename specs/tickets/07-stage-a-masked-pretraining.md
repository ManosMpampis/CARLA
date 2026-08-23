# 07: Two-stage learning part A — masked-block stage-A pretraining

**What to build:** The first learning stage: a MaskingCollator cuts contiguous blocks from windows and the dense latent-prediction objective learns to fill their latents — the model first learns how time series behave. Stage-A corpus selectable: `single` (one machine; official protocol) or `joint` (all SMD machines together; exploratory). PSM is always its own corpus.

**Blocked by:** 03 (real-data path), 04 (SIGReg arm).

**Status:** ready-for-agent

- [ ] Contiguous-block masking active only in stage-A configs; task training remains mask-free
- [ ] `corpus: single` trains on one machine's windows under the official per-machine protocol
- [ ] `corpus: joint` trains across all SMD machines' train splits in one run, with per-machine normalization respected
- [ ] Stage-A checkpoint persisted in the existing resume format as the handoff artifact for adaptation
