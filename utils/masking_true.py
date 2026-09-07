"""Input-space block masking for true-LeWM training.

Samples contiguous masked blocks on the INPUT axis and derives, per pyramid
level, the token mask of tokens whose covered input range overlaps a masked
block (via max-pooling, mirroring the encoder's cumulative strides).
Emits {"input": (B, W)} plus one {level: (B, T_l)} entry per level so the
model can mask the context stream in input space while the criterion
counts the loss on the matching latent tokens.
"""
import numpy as np
import torch
import torch.nn.functional as F


class InputBlockMaskCollator:
    """Contiguous-block input masking with stride-consistent token masks."""

    def __init__(self, num_blocks: int = 2, block_span: int = 32):
        self.num_blocks = int(num_blocks)
        self.block_span = int(block_span)

    def __call__(self, batch_size: int, window: int, level_strides: list) -> dict:
        input_mask = np.stack(
            [self._input_mask(window) for _ in range(batch_size)])
        masks: dict = {"input": torch.from_numpy(input_mask)}
        base = torch.from_numpy(input_mask).float().unsqueeze(1)  # (B,1,W)
        for idx, stride in enumerate(level_strides):
            s = max(int(stride), 1)
            tok = F.max_pool1d(base, kernel_size=s, stride=s).squeeze(1)
            masks[f"L{idx}"] = tok > 0.5
        return masks

    def _input_mask(self, window: int) -> np.ndarray:
        mask = np.zeros(window, dtype=bool)
        span = max(1, min(self.block_span, window))
        for _ in range(self.num_blocks):
            s = int(np.random.randint(0, window - span + 1))
            mask[s:s + span] = True
        if mask.all():  # keep at least some context visible
            mask[int(np.random.randint(0, window))] = False
        return mask
