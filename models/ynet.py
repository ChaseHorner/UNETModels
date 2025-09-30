import torch
import torch.nn as nn
import configs

from models.components.encoder import Encoder
from models.components.decoder import Decoder
from models.components.convblock import ConvBlock
from models.components.weather_compression import WeatherCompressionAvgPool

class Ynet(nn.Module):
    def __init__(
            self, 
            lidar_channels = configs.LIDAR_IN_CHANNELS, 
            sentinel_channels = configs.S1, 
            weather_channels = configs.WEATHER_IN_CHANNELS, 
            output_channels=1,
            config = configs
            ):

        super(Ynet, self).__init__()

        self.lidar_channels = lidar_channels
        self.sentinel_channels = sentinel_channels
        self.weather_channels = weather_channels
        self.output_channels = output_channels

        self.in_weather_in_season = WeatherCompressionAvgPool(weather_channels, config.W1, kernel_size=config.IN_SEASON_KERNEL_SIZE)
        self.in_weather_pre_season = WeatherCompressionAvgPool(weather_channels, config.W2, kernel_size=config.PRE_SEASON_KERNEL_SIZE)

        self.initial_conv = ConvBlock(lidar_channels, config.C0)
        self.enc_1 = Encoder(config.C0, config.C1)
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

        self.final_output = ConvBlock(config.C2 + config.S1, output_channels)

    def forward(self, **kwargs):
        x = kwargs.get('lidar')  # (b, lidar_channels, H, W) also called i1 or x1
        i2 = kwargs.get('sentinel')
        i3 = self.in_weather_in_season(kwargs.get('weather_in_season'))
        i4 = self.in_weather_pre_season(kwargs.get('weather_out_season'))

        x = self.initial_conv(x)
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