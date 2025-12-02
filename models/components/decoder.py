import torch
import torch.nn as nn
from models.components.convblock import ConvBlock

class Decoder(nn.Module):
    '''
    Decoder block that upsamples the input feature map and concatenates it with the corresponding skip connection from the encoder.
    It then applies a ConvBlock to refine the combined features.
    
    1. Upsampling Layer: Uses a transposed convolution (ConvTranspose2d) to upsample the input feature map by a specified scale size (default is 2 but depends on the corresponding encoder's scale size).
    2. Concatenation: Merges the upsampled feature map with the skip connection along the channel dimension.
    3. Convolutional Block: Applies a ConvBlock to the concatenated features to further process and refine them.

    This block takes input [batch_size, in_channels, height, width] 
        and a skip connection [batch_size, skip_channels, height*scale_size, width*scale_size], 
        and produces output [batch_size, out_channels, height*scale_size, width*scale_size].
    '''

    def __init__(self, in_channels, out_channels, skip_channels = 0, scale_size = 2):
        super(Decoder, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_size, stride=scale_size)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x = torch.concat([x, skip], dim=1)
        x = self.conv_block(x)
        return x