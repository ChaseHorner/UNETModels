import torch
from torch.utils.data import DataLoader
from chart_metrics import chart_metrics
from config_loader import configs
import models.unet as unet
import os
import torch.optim as optim
from train import train_model
from data_pipeline.data_loader import FieldDataset
from visualize_predictions import visualize_predictions
from objective_functions import *
from save_resfs import save_resfs
import time
import json
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_evaluate_model(model_name):
    # Load model status
    model_status_path = f'outputs/{model_name}/status.json'
    with open(model_status_path, "r") as f:
        model_status = json.load(f)


    # Skip if already finished
    if model_status["finished"]:
        return

    # Initialize model, optimizer, and data loaders
    train_dataset = FieldDataset(configs.DATASET_PATH, input_keys=configs.INPUT_KEYS, years=configs.TRAIN_YEARS)
    val_dataset = FieldDataset(configs.DATASET_PATH, input_keys=configs.INPUT_KEYS, years=configs.VAL_YEARS)

    train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

    unet_model = unet.Unet()
    unet_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  

    optimizer = optim.AdamW(unet_model.parameters(), lr=configs.LEARNING_RATE, betas=[configs.BETA1, 0.999])


    if model_status["model_path"] and model_status["optimizer_path"]:
        unet_model.load_state_dict(torch.load(model_status["model_path"]))
        optimizer.load_state_dict(torch.load(model_status["optimizer_path"]))


    metrics, model_path, optimizer_path, early_stopping = train_model(
        unet_model,
        configs.MODEL_NAME,
        configs.MODEL_FOLDER,
        optimizer,
        configs.CRITERION,
        train_loader,
        val_loader,
        configs.EPOCHS,
        device,
        start_epoch=model_status["last_trained_epoch"] + 1
    )

    model_status["metrics"] = metrics
    model_status["model_path"] = model_path
    model_status["optimizer_path"] = optimizer_path
    model_status["early_stopping"] = early_stopping
    model_status["last_trained_epoch"] += configs.EPOCHS
    model_status["last_trained_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


    if early_stopping:
        model_status["finished"] = True
        chart_metrics(metrics, configs.MODEL_FOLDER, configs.EPOCHS)
        visualize_predictions(unet_model, configs.MODEL_FOLDER, model_status["model_path"], val_dataset.with_field_year())
        save_resfs(configs.MODEL_FOLDER, configs.MODEL_NAME)

    # persist statuses: human-readable summary (JSON) + full state (torch)
    os.makedirs(configs.MODEL_FOLDER, exist_ok=True)


    json_path = os.path.join(configs.MODEL_FOLDER, f"status.json")
    with open(json_path, "w") as jf:
        json.dump(model_status, jf, default=str, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a model.')
    parser.add_argument('model_name', type=str, help='The name of the model to train.')
    args = parser.parse_args()
    
    train_and_evaluate_model(args.model_name)