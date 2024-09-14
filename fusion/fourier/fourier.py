import torch
import torch.nn as nn

from fusion.fourier.fc import MLP
from fusion.utils.model_cfgs import FusionConfigs


class FourierBlock(nn.Module):
    def __init__(self):
        super(FourierBlock, self).__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x

