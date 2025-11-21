import torch
import torch.nn as nn
from models.components.convblock import ConvBlock

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels = 0, scale_size = 2):
        super(Decoder, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_size, stride=scale_size)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x = torch.concat([x, skip], dim=1)
        x = self.conv_block(x)
        return x