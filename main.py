import torch
from torch.utils.data import DataLoader, random_split
from chart_metrics import chart_metrics
from config_loader import configs
import models.unet as unet
import os
import torch.optim as optim
from train import train_model
from data_pipeline.data_loader import FieldDataset
from visualize_predictions import visualize_predictions
from objective_functions import *


print("Imports complete")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = FieldDataset(configs.DATASET_PATH, input_keys=configs.INPUT_KEYS)
print(f"Dataset loaded with {len(dataset)} samples")

train_size = int(configs.TRAIN_VAL_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)
print(f"DataLoader created with {len(train_loader)} training batches and {len(val_loader)} validation batches")


unet_model = unet.Unet()
unet_model.to(device)  

optimizer = optim.AdamW(unet_model.parameters(), lr=configs.LEARNING_RATE, betas=[configs.BETA1, 0.999])

os.makedirs(configs.MODEL_FOLDER, exist_ok=True)

print("\n===============================\n")
print(f"Training {configs.MODEL_NAME} on {device} for {configs.EPOCHS} epochs...")
metrics = train_model(unet_model, 
                        configs.MODEL_NAME, 
                        configs.MODEL_FOLDER, 
                        optimizer, 
                        configs.CRITERION, 
                        train_loader, 
                        val_loader, 
                        configs.EPOCHS, 
                        device,
                        accu = True)


chart_metrics(metrics, configs.MODEL_FOLDER, configs.EPOCHS)

model_name = configs.MODEL_NAME + '_highest_psnr'
visualize_predictions(unet_model, configs.MODEL_FOLDER, model_name, dataset.with_field_year(), num_images=2)
