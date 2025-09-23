import torch
import torch.nn as nn
import configs

class WeatherCompression(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(WeatherCompression, self).__init__()
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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

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

class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(
            self, 
            lidar_channels = configs.LIDAR_IN_CHANNELS, 
            sentinel_channels = configs.SEN_IN_CHANNELS, 
            weather_channels = configs.WEATHER_IN_CHANNELS, 
            output_channels=1,
            config = configs
            ):

        super(Unet, self).__init__()

        self.lidar_channels = lidar_channels
        self.sentinel_channels = sentinel_channels
        self.weather_channels = weather_channels
        self.output_channels = output_channels

        self.in_weather_in_season = WeatherCompression(weather_channels, config.W1, kernel_size=config.IN_SEASON_KERNEL_SIZE)
        self.in_weather_pre_season = WeatherCompression(weather_channels, config.W2, kernel_size=config.PRE_SEASON_KERNEL_SIZE)

        self.enc_1 = Encoder(lidar_channels, config.C1)
        self.enc_2 = Encoder(config.C1, config.C2, scale_size=5)
        self.enc_3 = Encoder(config.C2 + config.S1, config.C3)
        self.enc_4 = Encoder(config.C3, config.C4)
        self.enc_5 = Encoder(config.C4, config.C5)
        self.enc_6 = Encoder(config.C5, config.C6)
        self.enc_7 = Encoder(config.C6, config.C7)

        self.dec_7 = Decoder(config.C7 + config.W1 + config.W2, config.C6, skip_channels=config.C6)
        self.dec_6 = Decoder(config.C6, config.C5, skip_channels=config.C5)
        self.dec_5 = Decoder(config.C5, config.C4, skip_channels=config.C4)
        self.dec_4 = Decoder(config.C4, config.C3, skip_channels=config.C3)
        self.dec_3 = Decoder(config.C3, config.C2 + config.S1, skip_channels=config.C2 + config.S1)

        self.final_output = FinalOutput(config.C3 + config.S1, output_channels)

    def forward(self, lidar_data, sentinel_data, weather_in_season_data, weather_out_season_data):
        x = lidar_data  # (b, lidar_channels, H, W) also called i1 or x1
        i2 = sentinel_data
        i3 = self.in_weather_in_season(weather_in_season_data)
        i4 = self.in_weather_pre_season(weather_out_season_data)

        x = self.enc_1(x)
        x = self.enc_2(x)

        x2 = torch.cat([x, i2], dim=1)
        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)
        x5 = self.enc_5(x4)
        x6 = self.enc_6(x5)
        x7 = self.enc_7(x6)

        i3 = i3.unsqueeze(-1).unsqueeze(-1)
        i4 = i4.unsqueeze(-1).unsqueeze(-1)
        i3 = i3.expand(-1, -1, x7.shape[2], x7.shape[3])
        i4 = i4.expand(-1, -1, x7.shape[2], x7.shape[3])
        x7 = torch.cat([x7, i3, i4], dim=1)

        x = self.dec_7(x7, x6)
        x = self.dec_6(x, x5)
        x = self.dec_5(x, x4)
        x = self.dec_4(x, x3)
        x = self.dec_3(x, x2)

        x = self.final_output(x)

        return x