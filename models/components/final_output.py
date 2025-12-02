import torch.nn as nn

class FinalOutput(nn.Module):
    '''
    Final output block that applies a 1x1 convolution to map the feature map to the desired number of output channels.

    This block takes input [batch_size, in_channels, height, width] 
        and produces output [batch_size, out_channels, height, width].
    '''

    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.conv(x)