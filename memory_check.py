import torch
import models.unet as unet
import models.unet4 as unet4
import models.unet16 as unet16
from config_loader import configs


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
