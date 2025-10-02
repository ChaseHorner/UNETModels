from models.components.weather_compression import WeatherCompression
from . import configs
import torch



def test_wc_in_season():
    in_channels = configs.WEATHER_IN_CHANNELS
    out_channels = configs.W1
    kernel_size = configs.IN_SEASON_KERNEL_SIZE
    input_length = configs.IN_SEASON_DAYS

    wc = WeatherCompression(in_channels, out_channels, kernel_size)
    input_tensor = torch.randn(1, in_channels, input_length)
    output_tensor = wc(input_tensor)


    assert output_tensor.shape == (1, out_channels), \
        f"Expected shape {(1, out_channels)}, but got {output_tensor.shape}"

def test_wc_pre_season():
    in_channels = configs.WEATHER_IN_CHANNELS
    out_channels = configs.W2
    kernel_size = configs.PRE_SEASON_KERNEL_SIZE
    input_length = configs.PRE_SEASON_DAYS

    wc = WeatherCompression(in_channels, out_channels, kernel_size)
    input_tensor = torch.randn(1, in_channels, input_length)
    output_tensor = wc(input_tensor)

    assert output_tensor.shape == (1, out_channels), \
        f"Expected shape {(1, out_channels)}, but got {output_tensor.shape}"
    

def test_wc_concat():
    in_channels = configs.WEATHER_IN_CHANNELS
    out_channels_1 = configs.W1
    out_channels_2 = configs.W2
    kernel_size_1 = configs.IN_SEASON_KERNEL_SIZE
    kernel_size_2 = configs.PRE_SEASON_KERNEL_SIZE
    input_length_1 = configs.IN_SEASON_DAYS
    input_length_2 = configs.PRE_SEASON_DAYS

    wc_1 = WeatherCompression(in_channels, out_channels_1, kernel_size_1)
    wc_2 = WeatherCompression(in_channels, out_channels_2, kernel_size_2)

    input_tensor_1 = torch.randn(1, in_channels, input_length_1)
    input_tensor_2 = torch.randn(1, in_channels, input_length_2)
    x7 = torch.randn(1, configs.C7, configs.BOTTLENECK_SIZE[0], configs.BOTTLENECK_SIZE[1])

    i3 = wc_1(input_tensor_1)
    i4 = wc_2(input_tensor_2)

    i3 = i3.unsqueeze(-1).unsqueeze(-1)
    i4 = i4.unsqueeze(-1).unsqueeze(-1)
    i3 = i3.expand(-1, -1, x7.shape[2], x7.shape[3])
    i4 = i4.expand(-1, -1, x7.shape[2], x7.shape[3])
    x7 = torch.cat([x7, i3, i4], dim=1)

    assert x7.shape == (1, configs.C7 + out_channels_1 + out_channels_2, configs.BOTTLENECK_SIZE[0], configs.BOTTLENECK_SIZE[1]), \
        f"Expected shape {(1, configs.C7 + out_channels_1 + out_channels_2, configs.BOTTLENECK_SIZE[0], configs.BOTTLENECK_SIZE[1])}, but got {x7.shape}"