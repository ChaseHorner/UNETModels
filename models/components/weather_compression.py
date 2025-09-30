import torch.nn as nn

class WeatherCompressionAvgPool(nn.Module):
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