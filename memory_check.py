import torch
import models.unet as unet
import models.unet4 as unet4
import models.unet16 as unet16
from config_loader import configs
from data_pipeline.data_loader import FieldDataset
from torch.utils.data import DataLoader


def memory_check():
    model = unet.Unet() if configs.BASE_MODEL == 'unet8' else \
            unet4.Unet4() if configs.BASE_MODEL == 'unet4' else \
            unet16.Unet16() if configs.BASE_MODEL == 'unet16'\
            else unet.Unet()

    inputs = {
        'lidar': torch.randn(1, configs.L1, 2560, 2560, device='cuda'),
        'sentinel': torch.randn(1, configs.S1, 256, 256, device='cuda')
    }


    model.to('cuda')

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model(**inputs)

    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak memory used: {peak:.2f} MB")


    print(f"{'Layer':50s} {'Param #':>12s} {'Shape':>20s}")
    print("-" * 90)

    total_params = 0
    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        print(f"{name:50s} {count:12,d} {str(list(param.shape)):>20s}")

    print("-" * 90)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Approx. model size: {total_params * 4 / (1024**3):.2f} GB (float32)")

def data_summary():
    #Not great, doesnt check each channel individually yet
    dataset = FieldDataset(configs.DATASET_PATH, input_keys=configs.INPUT_KEYS, years=configs.TRAIN_YEARS+configs.VAL_YEARS)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    #Find the min and max values for each input key
    input_mins = {key: float('inf') for key in configs.INPUT_KEYS}
    input_maxs = {key: float('-inf') for key in configs.INPUT_KEYS}

    for batch in dataloader:
        for key in configs.INPUT_KEYS:
            data = batch[key]
            input_mins[key] = min(input_mins[key], data.min().item())
            input_maxs[key] = max(input_maxs[key], data.max().item())

    print("Input data summary:")
    for key in configs.INPUT_KEYS:
        print(f"  {key}: min={input_mins[key]}, max={input_maxs[key]}")

if __name__ == "__main__":
    # memory_check()
    data_summary()