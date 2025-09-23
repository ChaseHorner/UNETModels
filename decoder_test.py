from .. import model
from .. import configs
import torch


def test_decoder_basic_scale_two():
    decoder = model.Decoder(64, 3, scale_size = 2)
    input_tensor = torch.randn(1, 64, 64, 64)
    skip_tensor = torch.randn(1, 0, 128, 128)  # No skip connection
    output_tensor = decoder(input_tensor, skip_tensor)

    assert output_tensor.shape == (1, 3, 128, 128), \
        f"Expected shape {(1, 3, 128, 128)}, but got {output_tensor.shape}"    
    
def test_decoder_start():
    in_channels = configs.C7 + configs.W1 + configs.W2
    out_channels = configs.C6

    decoder = model.Decoder(in_channels, out_channels, scale_size = 2)
    input_tensor = torch.randn(1, in_channels, 8, 8)
    skip_tensor = torch.randn(1, 0, 16, 16)  # No skip connection
    output_tensor = decoder(input_tensor, skip_tensor)

    assert output_tensor.shape == (1, out_channels, 16, 16), \
        f"Expected shape {(1, out_channels, 16, 16)}, but got {output_tensor.shape}"
    
def test_decoder_end():
    in_channels = configs.C3
    out_channels = configs.C2 + configs.S1
    skip_channels = configs.C2 + configs.S1

    decoder = model.Decoder(in_channels, out_channels, skip_channels=skip_channels, scale_size = 2)
    input_tensor = torch.randn(1, in_channels, 128, 128)
    skip_tensor = torch.randn(1, skip_channels, 256, 256)  # With skip connection
    output_tensor = decoder(input_tensor, skip_tensor)

    assert output_tensor.shape == (1, out_channels, 256, 256), \
        f"Expected shape {(1, out_channels, 256, 256)}, but got {output_tensor.shape}"

def test_decoder_chain_start():
    in_channels_1 = configs.C7 + configs.W1 + configs.W2
    out_channels_1 = configs.C6
    skip_channels_1 = configs.C6

    in_channels_2 = configs.C6
    out_channels_2 = configs.C5
    skip_channels_2 = configs.C5

    dec_1 = model.Decoder(in_channels_1, out_channels_1, skip_channels=skip_channels_1, scale_size=2)
    dec_2 = model.Decoder(in_channels_2, out_channels_2, skip_channels=skip_channels_2, scale_size=2)

    input_tensor = torch.randn(1, in_channels_1, 8, 8)
    skip_tensor_1 = torch.randn(1, skip_channels_1, 16, 16)
    output_tensor_1 = dec_1(input_tensor, skip_tensor_1)

    skip_tensor_2 = torch.randn(1, skip_channels_2, 32, 32)
    output_tensor_2 = dec_2(output_tensor_1, skip_tensor_2)

    assert output_tensor_2.shape == (1, out_channels_2, 32, 32), \
        f"Expected shape {(1, out_channels_2, 32, 32)}, but got {output_tensor_2.shape}"
    
def test_decoder_chain_full():
    dec_7 = model.Decoder(configs.C7 + configs.W1 + configs.W2, configs.C6, skip_channels=configs.C6)
    dec_6 = model.Decoder(configs.C6, configs.C5, skip_channels=configs.C5)
    dec_5 = model.Decoder(configs.C5, configs.C4, skip_channels=configs.C4)
    dec_4 = model.Decoder(configs.C4, configs.C3, skip_channels=configs.C3)
    dec_3 = model.Decoder(configs.C3, configs.C2 + configs.S1, skip_channels=configs.C2 + configs.S1)

    input_tensor = torch.randn(1, configs.C7 + configs.W1 + configs.W2, 8, 8)
    skip_tensor_7 = torch.randn(1, configs.C6, 16, 16)
    output_tensor_7 = dec_7(input_tensor, skip_tensor_7)

    skip_tensor_6 = torch.randn(1, configs.C5, 32, 32)
    output_tensor_6 = dec_6(output_tensor_7, skip_tensor_6)

    skip_tensor_5 = torch.randn(1, configs.C4, 64, 64)
    output_tensor_5 = dec_5(output_tensor_6, skip_tensor_5)

    skip_tensor_4 = torch.randn(1, configs.C3, 128, 128)
    output_tensor_4 = dec_4(output_tensor_5, skip_tensor_4)

    skip_tensor_3 = torch.randn(1, configs.C2 + configs.S1, 256, 256)
    output_tensor_3 = dec_3(output_tensor_4, skip_tensor_3)

    assert output_tensor_3.shape == (1, configs.C2 + configs.S1, 256, 256), \
        f"Expected shape {(1, configs.C2 + configs.S1, 256, 256)}, but got {output_tensor_3.shape}"