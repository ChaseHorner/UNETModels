import torch.nn as nn
from config_loader import configs

    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False),
            nn.InstanceNorm2d(out_channels) if configs.USE_IN_NORM else nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False),
            nn.InstanceNorm2d(out_channels) if configs.USE_IN_NORM else nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)