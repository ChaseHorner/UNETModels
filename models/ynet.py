import torch
import torch.nn as nn
from config_loader import configs

from models.components.encoder import Encoder
from models.components.decoder import Decoder
from models.components.convblock import ConvBlock
from models.components.weather_compression import WeatherCompressionAvgPool

class Ynet(nn.Module):
    '''
    **NOT USED YET**
    A Y-Net architecture with weather data integrated at the bottleneck.
    Needs C0-C7, W1-W2, S1, IN_SEASON_KERNEL_SIZE, and PRE_SEASON_KERNEL_SIZE defined in configs.
    '''

    def __init__(
            self, 
            lidar_channels = configs.LIDAR_IN_CHANNELS, 
            weather_channels = configs.WEATHER_IN_CHANNELS, 
            output_channels=1,
            config = configs
            ):

        super(Ynet, self).__init__()

        self.lidar_channels = lidar_channels
        self.weather_channels = weather_channels
        self.output_channels = output_channels

        #Define weather compression modules
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
        s2_data = kwargs.get('sentinel') #May be None
        #Handle reduced sentinel data
        if s2_data is None:
            s2_data = kwargs.get('s2_reduced')
            
        hmask = kwargs.get('hmask')
        auc = kwargs.get('auc') #May be None

        #Process weather data through compression modules
        in_weather = self.in_weather_in_season(kwargs.get('weather_in_season'))
        pre_weather = self.in_weather_pre_season(kwargs.get('weather_out_season'))

        x = self.initial_conv(x)
        x = self.enc_1(x)
        x = self.enc_2(x)

        #Concatenate all [256x256] data
        inputs = [x]
        for name in [s2_data, hmask, auc]:
            if name is not None:
                inputs.append(name)

        del s2_data, hmask, auc, x

        x2 = torch.cat(inputs, dim=1)
        del inputs

        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)
        x5 = self.enc_5(x4)
        x6 = self.enc_6(x5)
        x7 = self.enc_7(x6)

        #Expand weather data to match spatial dimensions and concatenate
        in_weather = in_weather.unsqueeze(-1).unsqueeze(-1)
        pre_weather = pre_weather.unsqueeze(-1).unsqueeze(-1)
        in_weather = in_weather.expand(-1, -1, x7.shape[2], x7.shape[3])
        pre_weather = pre_weather.expand(-1, -1, x7.shape[2], x7.shape[3])
        x7 = torch.cat([x7, in_weather, pre_weather], dim=1)
        del in_weather, pre_weather

        x = self.dec_7(x7, x6)
        del x7,  x6
        x = self.dec_6(x,  x5)
        del x5
        x = self.dec_5(x,  x4)
        del x4
        x = self.dec_4(x,  x3)
        del x3
        x = self.dec_3(x,  x2)
        del x2
        x = self.final_output(x)

        return x