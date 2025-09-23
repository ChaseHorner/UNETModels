from .. import model
from .. import configs
import torch



def test_encoder_basic_scale_two():
    encoder = model.Encoder(3, 64, scale_size = 2)
    input_tensor = torch.randn(1, 3, 128, 128)
    output_tensor = encoder(input_tensor)

    assert output_tensor.shape == (1, 64, 64, 64), \
        f"Expected shape {(1, 64, 64, 64)}, but got {output_tensor.shape}"


def test_encoder_basic_scale_five():
    encoder = model.Encoder(1, 32, scale_size = 5)
    input_tensor = torch.randn(1, 1, 100, 100)
    output_tensor = encoder(input_tensor)

    assert output_tensor.shape == (1, 32, 20, 20), \
        f"Expected shape {(1, 32, 20, 20)}, but got {output_tensor.shape}"

def test_encoder_with_sentinel_concat():
    scale_size = 2

    enc_1 = model.Encoder(configs.C2 + configs.S1, configs.C3, scale_size)

    input_tensor = torch.randn(1, configs.C2 + configs.S1, 256, 256)
    output_tensor = enc_1(input_tensor)

    assert output_tensor.shape == (1, configs.C3, 128, 128), \
        f"Expected shape {(1, configs.C3, 128, 128)}, but got {output_tensor.shape}"
    

## This one is really slow, so it's commented out for regular testing

def test_encoder_start():
    in_channels, height, width = configs.LIDAR_SIZE
    out_channels = configs.LIDAR_OUT_CHANNELS
    scale_size = 2
    batch_size = 1

    encoder = model.Encoder(in_channels, out_channels, scale_size)
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    output_tensor = encoder(input_tensor)

    expected_height = height // scale_size
    expected_width = width // scale_size

    assert output_tensor.shape == (batch_size, out_channels, expected_height, expected_width), \
        f"Expected shape {(batch_size, out_channels, expected_height, expected_width)}, but got {output_tensor.shape}"


def test_encoder_chain_end():
    enc_1 = model.Encoder(configs.C5, configs.C6, scale_size = 2)
    enc_2 = model.Encoder(configs.C6, configs.C7, scale_size = 2)

    input_tensor = torch.randn(1, configs.C5, 32, 32)
    output_tensor_1 = enc_1(input_tensor)
    output_tensor_2 = enc_2(output_tensor_1)

    assert output_tensor_2.shape == (1, configs.C7, 8, 8), \
        f"Expected shape {(1, configs.C7, 8, 8)}, but got {output_tensor_2.shape}"

def test_encoder_chain_start():
    enc_1 = model.Encoder(configs.C1, configs.C2, scale_size = 2)
    enc_2 = model.Encoder(configs.C2, configs.C3, scale_size = 5)

    input_tensor = torch.randn(1, configs.C1, 1280, 1280)
    output_tensor_1 = enc_1(input_tensor)
    output_tensor_2 = enc_2(output_tensor_1)

    assert output_tensor_2.shape == (1, configs.C3, 128, 128), \
        f"Expected shape {(1, configs.C3, 128, 128)}, but got {output_tensor_2.shape}"


def test_encoder_chain_sentinel_concat():
    enc_1 = model.Encoder(configs.C1, configs.C2, scale_size = 5)
    enc_2 = model.Encoder(configs.C2 + configs.S1, configs.C3, scale_size = 2)

    input_tensor = torch.randn(1, configs.C1, 1280, 1280)
    output_tensor_1 = enc_1(input_tensor)
    
    input_tensor_2 = torch.randn(1, configs.S1, 256, 256)
    output_tensor_1 = torch.cat((output_tensor_1, input_tensor_2), dim=1)
    output_tensor_2 = enc_2(output_tensor_1)

    assert output_tensor_2.shape == (1, configs.C3, 128, 128), \
        f"Expected shape {(1, configs.C3, 128, 128)}, but got {output_tensor_2.shape}"