import torch.nn as nn
from models.components.convblock import ConvBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale_size = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(scale_size),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x