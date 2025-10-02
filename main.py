import torch
from torch.utils.data import DataLoader, random_split
from chart_metrics import chart_metrics
import configs
import models.unet as unet
from torch import nn
import os
import torch.optim as optim
from train import train_model
from data_pipeline.data_loader import FieldDataset
from visualize_predictions import visualize_predictions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = FieldDataset("data/ten_samples", input_keys=configs.INPUT_KEYS)

train_size = int(configs.TRAIN_VAL_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

unet_model = unet.Unet()
unet_model.to(device)  

criterion = nn.L1Loss()
optimizer = optim.AdamW(unet_model.parameters(), lr=configs.LEARNING_RATE, betas=[configs.BETA1, 0.999])

os.makedirs(configs.MODEL_FOLDER, exist_ok=True)

print("\n===============================\n")
print(f"Training on {device} for {configs.EPOCHS} epochs...")
metrics = train_model(unet_model, 
                        configs.MODEL_NAME, 
                        configs.MODEL_FOLDER, 
                        optimizer, 
                        criterion, 
                        train_loader, 
                        val_loader, 
                        configs.EPOCHS, 
                        device)


chart_metrics(metrics, configs.MODEL_FOLDER, configs.EPOCHS)
visualize_predictions(unet_model, configs.MODEL_FOLDER, configs.MODEL_NAME, dataset.with_field_year(), num_images=2)
