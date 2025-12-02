import torch.nn as nn
from models.components.convblock import ConvBlock

class Encoder(nn.Module):
    '''
    Encoder block that downsamples the input feature map using MaxPooling followed by a ConvBlock.

    1. Downsampling Layer: Uses MaxPool2d to reduce the spatial dimensions of the input feature map by a specified scale size (default is 2).
    2. Convolutional Block: Applies a ConvBlock to the downsampled features to extract and refine features.

    This block takes input [batch_size, in_channels, height, width] 
        and produces output [batch_size, out_channels, height/scale_size, width/scale_size].
    '''

    def __init__(self, in_channels, out_channels, scale_size = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(scale_size),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x