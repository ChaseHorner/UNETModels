import torch
import torch.nn as nn
from models.components.convblock import ConvBlock

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels = 0, scale_size = 2):
        super(Decoder, self).__init__()

        self.conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale_size),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x = torch.concat([x, skip], dim=1)
        x = self.conv_block(x)
        return x