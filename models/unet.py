import torch
import torch.nn as nn
import configs

from models.components.encoder import Encoder
from models.components.decoder import Decoder
from models.components.final_output import FinalOutput

class Unet(nn.Module):
    def __init__(
            self, 
            lidar_channels = configs.LIDAR_IN_CHANNELS, 
            sentinel_channels = configs.S1, 
            output_channels=1,
            config = configs
            ):

        super(Unet, self).__init__()

        self.lidar_channels = lidar_channels
        self.sentinel_channels = sentinel_channels
        self.output_channels = output_channels

        self.enc_1 = Encoder(lidar_channels, config.C1)
        self.enc_2 = Encoder(config.C1, config.C2, scale_size=5)
        self.enc_3 = Encoder(config.C2 + config.S1, config.C3)
        self.enc_4 = Encoder(config.C3, config.C4)
        self.enc_5 = Encoder(config.C4, config.C5)
        self.enc_6 = Encoder(config.C5, config.C6)
        self.enc_7 = Encoder(config.C6, config.C7)

        self.dec_7 = Decoder(config.C7, config.C6, skip_channels=config.C6)
        self.dec_6 = Decoder(config.C6, config.C5, skip_channels=config.C5)
        self.dec_5 = Decoder(config.C5, config.C4, skip_channels=config.C4)
        self.dec_4 = Decoder(config.C4, config.C3, skip_channels=config.C3)
        self.dec_3 = Decoder(config.C3, config.C2 + config.S1, skip_channels=config.C2 + config.S1)

        self.final_output = FinalOutput(config.C2 + config.S1, output_channels)

    def forward(self, lidar_data, sentinel_data):
        x = lidar_data  # (b, lidar_channels, H, W) also called i1 or x1
        i2 = sentinel_data

        x = self.enc_1(x)
        x = self.enc_2(x)

        x2 = torch.cat([x, i2], dim=1)
        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)
        x5 = self.enc_5(x4)
        x6 = self.enc_6(x5)
        x7 = self.enc_7(x6)

        x = self.dec_7(x7, x6)
        x = self.dec_6(x, x5)
        x = self.dec_5(x, x4)
        x = self.dec_4(x, x3)
        x = self.dec_3(x, x2)

        x = self.final_output(x)

        return x