import numpy as np
import torch


class MaskingCollator:
    """Contiguous-block masking for stage-A latent prediction.

    Samples blocks on the *input* axis and marks the pyramid-level tokens
    whose covered input range overlaps them. Produces one boolean mask per
    batch sample per level: {level: (B, T_l)}. V-JEPA block-sampler
    mechanics flattened to 1D: block starts are drawn uniformly, blocks may
    not extend past the window, and a minimum number of visible context
    tokens is guaranteed at every level.
    """

    def __init__(self, num_blocks: int = 4, block_span: int = 24,
                 min_context_tokens: int = 2):
        self.num_blocks = num_blocks
        self.block_span = block_span
        self.min_context_tokens = min_context_tokens

    def __call__(self, batch_size: int, window: int, level_strides: list) -> dict:
        masks = {}
        for level_idx, stride in enumerate(level_strides):
            n_tokens = window // stride
            if n_tokens <= self.min_context_tokens:
                continue  # too coarse to mask meaningfully; stays fully visible
            per_sample = np.stack([
                self._token_mask(n_tokens, stride) for _ in range(batch_size)
            ])
            masks[f"L{level_idx}"] = torch.from_numpy(per_sample)
        return masks

    def _token_mask(self, n_tokens: int, stride: int) -> np.ndarray:
        # Sample blocks directly in token space of this level so that each
        # level sees the same *fraction* of masked span as the input axis.
        mask = np.zeros(n_tokens, dtype=bool)
        span = max(1, self.block_span // stride)
        for _ in range(self.num_blocks):
            s = int(np.random.randint(0, n_tokens - span + 1))
            mask[s:s + span] = True
        if mask.all():  # keep at least some context tokens visible
            visible = int(np.random.randint(0, n_tokens))
            mask[visible] = False
        return mask
