import time
from flask import json
import torch
from config_loader import configs
from objective_functions import *

def train_epoch(model, optimizer, criterion, train_dataloader, device):
    model.train()
    running_MSE, running_MAE, running_SSIM, total_count = 0.0, 0.0, 0.0, 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(train_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        target = inputs.pop("target")
        predictions = model(**inputs)

        count = (inputs.get('hmask') == 1.0).sum().item()
        loss = criterion(predictions, target, inputs.get('hmask'))
        (loss * count / configs.ACCUMULATION_STEPS).backward()

        if (step + 1) % configs.ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        
        running_MSE += MSE()(predictions, target, inputs.get('hmask')).item() * count
        running_MAE += MAE()(predictions, target, inputs.get('hmask')).item() * count
        running_SSIM += CroppedSSIM()(predictions, target, inputs.get('hmask')).item() * count
        total_count += count

        print(f"Training Step [{step+1}/{len(train_dataloader)}]", end="\r")

    if (step + 1) % configs.ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Compute the average loss across all pixels seen in the epoch
    return {
        "MSE": running_MSE / total_count,
        "RMSE": (running_MSE / total_count) ** 0.5,
        "MAE": running_MAE / total_count,
        "SSIM": running_SSIM / total_count,
    }



def evaluate_epoch(model, valid_dataloader, device):
    model.eval()
    total_MSE, total_MAE, total_SSIM, total_count = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in valid_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}

            target = inputs.pop("target")
            predictions = model(**inputs) 
        
            count = (inputs.get('hmask') == 1.0).sum().item()

            total_MSE += MSE()(predictions, target, inputs.get('hmask')).item() * count
            total_MAE += MAE()(predictions, target, inputs.get('hmask')).item() * count
            total_SSIM += CroppedSSIM()(predictions, target, inputs.get('hmask')).item() * count
            total_count += count

    return {
        "MSE": total_MSE / total_count,
        "RMSE": (total_MSE / total_count) ** 0.5,
        "MAE": total_MAE / total_count,
        "SSIM": total_SSIM / total_count,
    }

def train_model(model, model_name, model_folder, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device):
    train_mses, train_rmses, train_maes, train_ssims = [], [], [], []
    eval_mses, eval_rmses, eval_maes, eval_ssims = [], [], [], []
    best_mse_eval = (float('inf'), -1)  # (loss, epoch)
    best_mae_eval = (float('inf'), -1)
    best_ssim_eval = (-float('inf'), -1)
    times = []

    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Training
        train_metrics = train_epoch(model, optimizer, criterion, train_dataloader, device)
        train_mses.append(to_float(train_metrics["MSE"]))
        train_rmses.append(to_float(train_metrics["RMSE"]))
        train_maes.append(to_float(train_metrics["MAE"]))
        train_ssims.append(to_float(train_metrics["SSIM"]))

        # Evaluation
        eval_metrics = evaluate_epoch(model, valid_dataloader, device)
        eval_mses.append(to_float(eval_metrics["MSE"]))
        eval_rmses.append(to_float(eval_metrics["RMSE"]))
        eval_maes.append(to_float(eval_metrics["MAE"]))
        eval_ssims.append(to_float(eval_metrics["SSIM"]))

        # Save best model based on eval loss
        if best_mse_eval[0] > eval_metrics["MSE"]:
            torch.save(model.state_dict(), model_folder + f'/{model_name}_lowest_mse.pt')
            best_mse_eval = (eval_metrics["MSE"], epoch)
        # Save best model based on eval MAE
        if best_mae_eval[0] > eval_metrics["MAE"]:
            torch.save(model.state_dict(), model_folder + f'/{model_name}_lowest_mae.pt')
            best_mae_eval = (eval_metrics["MAE"], epoch)
        # Save best model based on eval SSIM
        if best_ssim_eval[0] < eval_metrics["SSIM"]:
            torch.save(model.state_dict(), model_folder + f'/{model_name}_highest_ssim.pt')
            best_ssim_eval = (eval_metrics["SSIM"], epoch)

        # Print and log loss at end of epochs
        with open(f"{model_folder}/logs.txt", "a") as f:
            f.write("-" * 59 + "\n")
            f.write(
                "| End of epoch {:3d} | Time: {:5.2f}s | Train MSE {:8.3f} | Train RMSE {:8.3f} | Train MAE {:8.3f} | Train SSIM {:8.3f} "
                "| Eval MSE {:8.3f} | Eval RMSE {:8.3f} | Eval MAE {:8.3f} | Eval SSIM {:8.3f} ".format(
                    epoch, time.time() - epoch_start_time, train_metrics["MSE"], train_metrics["RMSE"], train_metrics["MAE"], train_metrics["SSIM"],
                    eval_metrics["MSE"], eval_metrics["RMSE"], eval_metrics["MAE"], eval_metrics["SSIM"]
                )
                + "\n"
            )
            f.write("-" * 59 + "\n")

        print("-" * 59)
        print("| End of epoch {:3d} | Time: {:5.2f}s | Train MSE {:8.3f} | Train RMSE {:8.3f} | Train MAE {:8.3f} | Train SSIM {:8.3f} "
              "| Eval MSE {:8.3f} | Eval RMSE {:8.3f} | Eval MAE {:8.3f} | Eval SSIM {:8.3f} ".format(
                  epoch, time.time() - epoch_start_time, train_metrics["MSE"], train_metrics["RMSE"], train_metrics["MAE"], train_metrics["SSIM"],
                  eval_metrics["MSE"], eval_metrics["RMSE"], eval_metrics["MAE"], eval_metrics["SSIM"]
              )
        )
        print("-" * 59)
    
    # Save epoch number to a txt file
    with open(f"{model_folder}/{model_name}_saved_epochs.txt", "a") as f:
        f.write(f"Best MSE Epoch: {best_mse_eval[1]} with MSE: {best_mse_eval[0]:.4f}\n")
        f.write(f"Best MAE Epoch: {best_mae_eval[1]} with MAE: {best_mae_eval[0]:.4f}\n")
        f.write(f"Best SSIM Epoch: {best_ssim_eval[1]} with SSIM: {best_ssim_eval[0]:.4f}\n")

    metrics = {
        'train_mse': train_mses,
        'train_rmse': train_rmses,
        'train_mae': train_maes,
        'train_ssim': train_ssims,
        'eval_mse': eval_mses,
        'eval_rmse': eval_rmses,
        'eval_mae': eval_maes,
        'eval_ssim': eval_ssims,
        'time': times
    }

    # Save metrics to a JSON file
    with open(f"{model_folder}/metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


def to_float(x):
    return x.item() if torch.is_tensor(x) else float(x)