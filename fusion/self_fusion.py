import torch
import torch.nn as nn

from fusion.utils.layer_norm import LayerNorm
from fusion.fourier.fourier import FourierBlock
from fusion.utils.model_cfgs import FusionConfigs

class FourierSelfFusion(nn.Module):
    def __init__(self, using_ff_residual):
        super(FourierSelfFusion, self).__init__()
        fourier_config = FusionConfigs.FourierConfigs()
        self.fn = FourierBlock()
        self.norm1 = LayerNorm(size=fourier_config.HIDDEN_SIZE)

    def forward(self, y):
        y = self.norm1(
            y + self.fn(y)
        )
        return y
