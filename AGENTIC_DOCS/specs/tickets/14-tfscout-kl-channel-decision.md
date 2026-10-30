# 14: KL channel view-disagreement vs window-VAE fallback

**What to build:** Per-token symmetric view disagreement as the default distribution-alignment score channel, with window-level variational fallback and gated per-level tiny latents, so sub-window disagreement is tested with a collapse-safe alternative.

**Blocked by:** 13 (trunk plus open-loop baseline).

**Status:** ready-for-agent

- [ ] View-disagreement channel trains with asymmetry guard plus variance guard and emits a per-position map
- [ ] Fallback variational objective trains with annealing schedule and emits residual-style scores
- [ ] Per-level tiny latents stay behind annealing plus free-bits gating
- [ ] Tournament promotes one alignment tactic on honest metrics over the no-training baseline
