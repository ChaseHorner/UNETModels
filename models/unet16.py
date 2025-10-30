import torch
import torch.nn as nn
from config_loader import configs

from models.components.encoder import Encoder
from models.components.decoder import Decoder
from models.components.convblock import ConvBlock

class Unet16(nn.Module):
    def __init__(
            self, 
            lidar_channels = configs.LIDAR_IN_CHANNELS, 
            output_channels=1,
            config = configs
            ):

        super(Unet16, self).__init__()

        self.lidar_channels = lidar_channels
        self.output_channels = output_channels

        self.initial_conv = ConvBlock(lidar_channels, config.C0)    
        self.enc_1 = Encoder(config.C0, config.C1)
        self.enc_2 = Encoder(config.C1, config.C2, scale_size=5)
        self.enc_3 = Encoder(config.C2 + config.S1, config.C3)
        self.enc_4 = Encoder(config.C3, config.C4)
        self.enc_5 = Encoder(config.C4, config.C5)
        self.enc_6 = Encoder(config.C5, config.C6)

        self.dec_6 = Decoder(config.C6, config.C5, skip_channels=config.C5)
        self.dec_5 = Decoder(config.C5, config.C4, skip_channels=config.C4)
        self.dec_4 = Decoder(config.C4, config.C3, skip_channels=config.C3)
        self.dec_3 = Decoder(config.C3, config.C2 + config.S1, skip_channels=config.C2 + config.S1)

        self.final_output = ConvBlock(config.C2 + config.S1, output_channels)

    def forward(self, **kwargs):
        x = kwargs.get('lidar')  # (b, lidar_channels, H, W) also called i1 or x1
        s2_data = kwargs.get('sentinel')
        hmsk = kwargs.get('hmask')

        x = self.initial_conv(x)
        x = self.enc_1(x)
        x = self.enc_2(x)

        x2 = torch.cat([x, s2_data, hmsk], dim=1)
        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)
        x5 = self.enc_5(x4)
        x6 = self.enc_6(x5)

        x = self.dec_6(x6, x5)
        x = self.dec_5(x, x4)
        x = self.dec_4(x, x3)
        x = self.dec_3(x, x2)

        x = self.final_output(x)

        return x