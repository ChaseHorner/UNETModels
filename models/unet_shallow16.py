import torch
import torch.nn as nn
from config_loader import configs

from models.components.encoder import Encoder
from models.components.decoder import Decoder
from models.components.convblock import ConvBlock

class UnetShallow16(nn.Module):
    '''
    A U-Net architecture with a bottleneck of [16x16] and scales down by factors of 2,5,4,4 in the encoder.
    Needs C0-C4 and S1 defined in configs.
    '''
    def __init__(
            self, 
            lidar_channels = configs.LIDAR_IN_CHANNELS, 
            output_channels=1,
            config = configs
            ):

        super(UnetShallow16, self).__init__()

        self.lidar_channels = lidar_channels
        self.output_channels = output_channels

        self.initial_conv = ConvBlock(lidar_channels, config.C0)    
        self.enc_1 = Encoder(config.C0, config.C1)
        self.enc_2 = Encoder(config.C1, config.C2, scale_size=5)
        self.enc_3 = Encoder(config.C2 + config.S1, config.C3, scale_size=4)
        self.enc_4 = Encoder(config.C3, config.C4, scale_size=4)

        self.dec_4 = Decoder(config.C4, config.C3, skip_channels=config.C3, scale_size=4)
        self.dec_3 = Decoder(config.C3, config.C2 + config.S1, skip_channels=config.C2 + config.S1, scale_size=4)

        self.final_output = ConvBlock(config.C2 + config.S1, output_channels)

    def forward(self, **kwargs):
        x = kwargs.get('lidar')  # (b, lidar_channels, H, W) also called i1 or x1
        s2_data = kwargs.get('sentinel') #May be None
        
        #Handle reduced sentinel data
        if s2_data is None:
            s2_data = kwargs.get('s2_reduced')
        hmsk = kwargs.get('hmask')
        auc = kwargs.get('auc') #May be None

        x = self.initial_conv(x)
        x = self.enc_1(x)
        x = self.enc_2(x)

        #Concatenate all [256x256] data 
        inputs = [x]
        for name in [s2_data, hmsk, auc]:
            if name is not None:
                inputs.append(name)
        del s2_data, hmsk, auc, x

        x2 = torch.cat(inputs, dim=1)
        del inputs
        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)


        x = self.dec_4(x4, x3)
        del x3
        x = self.dec_3(x, x2)
        del x2

        x = self.final_output(x)
        torch.cuda.empty_cache() 


        return x