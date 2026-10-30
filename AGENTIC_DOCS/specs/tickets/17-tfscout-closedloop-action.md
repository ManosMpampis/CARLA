# 17: Closed-loop action-conditioned variant

**What to build:** Owner proposition arm where detached interval proposals feed an action embedding that conditionally modulates predictor and decoder, tested strictly against the open-loop winner, so feedback conditioning is proven rather than assumed.

**Blocked by:** 13 (trunk plus open-loop baseline), 14 (KL channel decision), 16 (shape-loss ablation).

**Status:** ready-for-agent

- [ ] Proposals detach before action embedding so no gradient loop into the detector
- [ ] Conditional modulation starts near identity and trains on a separate schedule
- [ ] Gradient-conflict monitoring with fallback projection when main and auxiliary gradients oppose
- [ ] Variant promotes only by beating open-loop winner plus no-training baseline on honest metrics
