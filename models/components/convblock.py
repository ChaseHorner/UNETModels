import torch.nn as nn
from config_loader import configs

    
class ConvBlock(nn.Module):
    '''
    A convolutional block that consists of two convolutional layers, each followed by normalization and LeakyReLU activation.
    1. Convolutional Layer: Applies a 2D convolution with specified input and output channels, kernel size (default 3), stride of 1, and padding to maintain spatial dimensions.
    2. Normalization Layer: Depending on the configuration, it applies either Instance Normalization or Batch Normalization (default) to stabilize and accelerate training.
    3. Activation Layer: Uses LeakyReLU activation function to introduce non-linearity

    This block takes input [batch_size, in_channels, height, width] 
        and produces output [batch_size, out_channels, height, width].

    It is the base block used in the encoder and decoder
    '''
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