import copy

import torch
import torch.nn as nn


class EMAWrapper(nn.Module):
    """Exponential-moving-average target encoder (V-JEPA-style teacher).

    Holds a frozen copy of the online encoder whose weights trail the
    online weights: after every optimizer step the trainer calls
    ``update()`` and each target parameter moves toward the online one
    with momentum m. Targets are produced under stop-gradient by
    construction (params have requires_grad=False and are never in the
    optimization graph).
    """

    def __init__(self, encoder: nn.Module, momentum: float = 0.99925):
        super().__init__()
        self.momentum = float(momentum)
        self.encoder = copy.deepcopy(encoder)
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        m = self.momentum
        t_params = self.encoder.state_dict()
        for name, param in online_encoder.state_dict().items():
            buf = t_params[name]
            if torch.is_floating_point(buf):
                buf.mul_(m).add_(param.detach(), alpha=1.0 - m)
            else:
                buf.copy_(param)
        self.encoder.load_state_dict(t_params)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> dict:
        return self.encoder(x)

    def train_mode(self) -> None:
        # Target stays in eval mode always: its BatchNorm statistics must
        # not track training batches (teacher statistics are part of EMI).
        self.encoder.eval()

    def train(self, mode: bool = True):
        # Keep the inner encoder in eval mode even when JEPAModel.train()
        # is called on the whole module.
        super().train(mode)
        self.encoder.eval()
        return self
