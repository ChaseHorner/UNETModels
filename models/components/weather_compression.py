import torch.nn as nn

'''
The difference between WeatherCompressionAvgPool and WeatherCompressionMaxPool is the type of pooling operation used to compress the temporal dimension after the convolutional layers.
Choose WeatherCompressionAvgPool when you want to retain the average characteristics of the time series data, which is useful for smoothing and reducing noise.
Choose WeatherCompressionMaxPool when you want to retain the most prominent features of the time series data, which is useful for capturing peak values and significant events.
'''

class WeatherCompressionAvgPool(nn.Module):
    '''
    Weather compression block that applies two 1D convolutions followed by an adaptive average pooling to compress the temporal dimension.
    1. Convolutional Layers: Applies two 1D convolutions with specified input and output channels, kernel size, and padding to maintain temporal dimensions.
    2. Activation Layers: Uses LeakyReLU activation function to introduce non-linearity
    3. Adaptive Average Pooling: Reduces the temporal dimension to 1, effectively compressing the time series data.

    This block takes input [batch_size, in_channels, time_steps] 
        and produces output [batch_size, out_channels].
    '''
    def __init__(self, in_channels, out_channels, kernel_size):
        super(WeatherCompressionAvgPool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1) # (b, out_channels, ~ts_len) -> (b, out_channels, ts_len/2)
        )

    def forward(self, x):
        h = self.conv(x).squeeze(-1) #(b, out_channels, 1) -> (b, out_channels)
        return h


class WeatherCompressionMaxPool(nn.Module):
    '''
    Weather compression block that applies two 1D convolutions followed by an adaptive max pooling to compress the temporal dimension.
    1. Convolutional Layers: Applies two 1D convolutions with specified input and output channels, kernel size, and padding to maintain temporal dimensions.
    2. Activation Layers: Uses LeakyReLU activation function to introduce non-linearity
    3. Adaptive Max Pooling: Reduces the temporal dimension to 1, effectively compressing the time series data.
    
    This block takes input [batch_size, in_channels, time_steps] 
        and produces output [batch_size, out_channels].
    '''

    def __init__(self, in_channels, out_channels, kernel_size):
        super(WeatherCompressionMaxPool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool1d(1) # (b, out_channels, ~ts_len) -> (b, out_channels, ts_len/2)
        )

    def forward(self, x):
        h = self.conv(x).squeeze(-1) #(b, out_channels, 1) -> (b, out_channels)
        return h