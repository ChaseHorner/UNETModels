import torch
from torch.utils.data import DataLoader
import configs
import models.unet as unet
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import os
import torch.optim as optim
from train_unet import train_model
from data_loader import FieldDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = FieldDataset("data/ten_samples", input_keys=['lidar', 'sentinel'])


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

unet_model = unet.Unet()
unet_model.to(device)  

criterion = nn.L1Loss()
optimizer = optim.AdamW(unet_model.parameters(), lr=1e-4, betas=[0.5,0.999])
EPOCHS = 40

save_model = './UNET'
os.makedirs(save_model, exist_ok = True)
MODEL_NAME = 'unet_model_2'

print("\n===============================\n")
print(f"Training on {device} for {EPOCHS} epochs...")
unet_model, metrics = train_model(
    unet_model, MODEL_NAME, save_model, optimizer, criterion, train_loader, val_loader, EPOCHS, device
)

