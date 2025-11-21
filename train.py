import time
from flask import json
import torch
from config_loader import configs
from objective_functions import *
import os


def train_epoch(model, optimizer, criterion, train_dataloader, device):
    '''Train the model for one epoch.'''
    
    model.train()

    # Track running metrics
    running_MSE, running_MAE, running_SSIM, total_count = 0.0, 0.0, 0.0, 0.0

    # Reset gradients at the start of the epoch
    optimizer.zero_grad()

    # Loop over batches
    for step, batch in enumerate(train_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        target = inputs.pop("target")

        predictions = model(**inputs)
        loss = criterion(predictions, target, inputs.get('hmask'))


        # Compute loss and backpropagate
        count = (inputs.get('hmask') == 1.0).sum().item()
        (loss * count / configs.ACCUMULATION_STEPS).backward()

        # Update weights after accumulation steps
        if (step + 1) % configs.ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update running metrics
        running_MSE += MSE()(predictions, target, inputs.get('hmask')).item() * count
        running_MAE += MAE()(predictions, target, inputs.get('hmask')).item() * count
        running_SSIM += CroppedSSIM()(predictions, target, inputs.get('hmask')).item() * count
        total_count += count

        print(f"Training Step [{step+1}/{len(train_dataloader)}]", end="\r")

    # Final weight update if there are remaining gradients
    if (step + 1) % configs.ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.empty_cache() 

    # Compute the average loss across all pixels seen in the epoch
    return {
        "MSE": running_MSE / total_count,
        "RMSE": (running_MSE / total_count) ** 0.5,
        "MAE": running_MAE / total_count,
        "SSIM": running_SSIM / total_count,
    }


def evaluate_epoch(model, valid_dataloader, device):
    '''Evaluate the model for one epoch.'''

    model.eval()
    
    # Track running metrics
    total_MSE, total_MAE, total_SSIM, total_count = 0.0, 0.0, 0.0, 0.0
    total_bpa_MSE, total_bpa_MAE = 0.0, 0.0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Loop over batches
        for batch in valid_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}

            target = inputs.pop("target")
            predictions = model(**inputs) 
        
            # Update running metrics
            count = (inputs.get('hmask') == 1.0).sum().item()

            total_MSE += MSE()(predictions, target, inputs.get('hmask')).item() * count
            total_MAE += MAE()(predictions, target, inputs.get('hmask')).item() * count
            total_SSIM += CroppedSSIM()(predictions, target, inputs.get('hmask')).item() * count
            total_count += count

            bpa_MSE, bpa_MAE = BushelsPerAcreErrors()(predictions, target, inputs.get('hmask'))
            total_bpa_MSE += bpa_MSE
            total_bpa_MAE += bpa_MAE

    # Compute the average loss across all pixels seen in the epoch
    return {
        "MSE": total_MSE / total_count,
        "RMSE": (total_MSE / total_count) ** 0.5,
        "MAE": total_MAE / total_count,
        "SSIM": total_SSIM / total_count,
        "bpa_MSE": total_bpa_MSE / len(valid_dataloader.dataset),
        "bpa_RMSE": (total_bpa_MSE / len(valid_dataloader.dataset)) ** 0.5,
        "bpa_MAE": total_bpa_MAE / len(valid_dataloader.dataset)
    }

def train_model(model, model_name, model_folder, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device, start_epoch=1, metrics=None):
    if metrics is None:
        metrics = {
            "train_mses": [],
            "train_rmses": [],
            "train_maes": [],
            "train_ssims": [],
            "eval_mses": [],
            "eval_rmses": [],
            "eval_maes": [],
            "eval_ssims": [],
            "bpa_MSEs": [],
            "bpa_RMSEs": [],
            "bpa_MAEs": [],
        }
    train_mses, train_rmses, train_maes, train_ssims= [], [], [], []
    eval_mses, eval_rmses, eval_maes, eval_ssims = [], [], [], [],
    bpa_MSEs, bpa_RMSEs, bpa_MAEs = [], [], []
    model_path = model_folder + f'/{model_name}_best.pt'
    optimizer_path = model_folder + f'/{model_name}_optimizer_best.pt'   
    early_stopping = False
    last_epoch = start_epoch + num_epochs - 1
    best_epoch = {'epoch': -1, 'MSE': float('inf'), 'MAE': float('inf'), 'SSIM': float('-inf'), 'bpa_MSE': float('inf'), 'bpa_RMSE': float('inf'), 'bpa_MAE': float('inf')}

    train_start_time = time.time()
    for epoch in range(start_epoch, start_epoch + num_epochs):

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

        bpa_MSEs.append(to_float(eval_metrics["bpa_MSE"]))
        bpa_RMSEs.append(to_float(eval_metrics["bpa_RMSE"]))
        bpa_MAEs.append(to_float(eval_metrics["bpa_MAE"]))

        # Save best model based on eval loss
        if best_epoch['MSE'] > eval_metrics["MSE"]:
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            best_epoch['epoch'] = epoch
            best_epoch['MSE'] = eval_metrics["MSE"]
            best_epoch['MAE'] = eval_metrics["MAE"]
            best_epoch['SSIM'] = eval_metrics["SSIM"]
            best_epoch['bpa_MSE'] = eval_metrics["bpa_MSE"]
            best_epoch['bpa_RMSE'] = eval_metrics["bpa_RMSE"]
            best_epoch['bpa_MAE'] = eval_metrics["bpa_MAE"]

        # Print and log loss at end of epochs
        with open(f"{model_folder}/logs.txt", "a") as f:
            f.write("-" * 59 + "\n")
            f.write(
                "| End of epoch {:3d} | Time: {:5.2f}s | Train MSE {:8.3f} | Train RMSE {:8.3f} | Train MAE {:8.3f} | Train SSIM {:8.3f} |"
                "| Eval MSE {:8.3f} | Eval RMSE {:8.3f} | Eval MAE {:8.3f} | Eval SSIM {:8.3f} | bpa MSE {:8.3f} | bpa RMSE {:8.3f} | bpa MAE {:8.3f} |".format(
                    epoch, time.time() - epoch_start_time, train_metrics["MSE"], train_metrics["RMSE"], train_metrics["MAE"], train_metrics["SSIM"],
                    eval_metrics["MSE"], eval_metrics["RMSE"], eval_metrics["MAE"], eval_metrics["SSIM"],
                    eval_metrics["bpa_MSE"], eval_metrics["bpa_RMSE"], eval_metrics["bpa_MAE"]
                )
                + "\n"
            )
            f.write("-" * 59 + "\n")

        print("-" * 59)
        print("| End of epoch {:3d} | Time: {:5.2f}s | Train MSE {:8.3f} | Train RMSE {:8.3f} | Train MAE {:8.3f} | Train SSIM {:8.3f} |"
                "| Eval MSE {:8.3f} | Eval RMSE {:8.3f} | Eval MAE {:8.3f} | Eval SSIM {:8.3f} | bpa MSE {:8.3f} | bpa RMSE {:8.3f} | bpa MAE {:8.3f} |".format(
                    epoch, time.time() - epoch_start_time, train_metrics["MSE"], train_metrics["RMSE"], train_metrics["MAE"], train_metrics["SSIM"],
                    eval_metrics["MSE"], eval_metrics["RMSE"], eval_metrics["MAE"], eval_metrics["SSIM"],
                    eval_metrics["bpa_MSE"], eval_metrics["bpa_RMSE"], eval_metrics["bpa_MAE"]
                )
        )
        print("-" * 59)

        # Early stopping based on eval MSE
        if eval_metrics["MSE"] > 1.05 * best_epoch['MSE']:
            print(f"Evaluating stopping at epoch {epoch} due to increasing eval MSE.")
            if early_stop(eval_mses, configs.EARLY_STOPPING_LENGTH, configs.EARLY_STOPPING_THRESHOLD):
                early_stopping = True
                last_epoch = epoch
                print(f"Early stopping triggered at epoch {epoch}.")
                with open(f"{model_folder}/logs.txt", "a") as f:
                    f.write(f"Early stopping triggered at epoch {epoch}.\n")
                break

        # Stop training if total wall-clock time exceeds 5 hours
        elapsed_hours = (time.time() - train_start_time) / 3600.0
        if elapsed_hours >= 5.75:
            print(f"Stopping training after {elapsed_hours:.2f} hours (limit: 5.75 hours).")
            with open(f"{model_folder}/logs.txt", "a") as f:
                f.write(f"Stopped training after {elapsed_hours:.2f} hours (limit: 5.75 hours).\n")
            break
    
    print(f"Training completed in {time.time() - train_start_time:.2f} seconds.")
    if best_epoch['epoch'] != (start_epoch + num_epochs):
        torch.save(model.state_dict(), model_folder + f'/{model_name}_{num_epochs}.pt')
        torch.save(optimizer.state_dict(), model_folder + f'/{model_name}_optimizer_{num_epochs}.pt')

        try:
            if best_epoch['epoch'] > 0:
                if os.path.exists(model_path):
                    new_model_path = os.path.join(model_folder, f"{model_name}_best_epoch{best_epoch['epoch']}.pt")
                    os.replace(model_path, new_model_path)
                    model_path = new_model_path
                if os.path.exists(optimizer_path):
                    new_optimizer_path = os.path.join(model_folder, f"{model_name}_optimizer_best_epoch{best_epoch['epoch']}.pt")
                    os.replace(optimizer_path, new_optimizer_path)
                    optimizer_path = new_optimizer_path
        except Exception as e:
            print(f"Failed to rename best model files: {e}")

    # Save epoch number to a txt file
    with open(f"{model_folder}/{model_name}_saved_epochs.txt", "a") as f:
        f.write(f"Best Epoch: {best_epoch['epoch']} with MSE: {best_epoch['MSE']:.4f} | MAE: {best_epoch['MAE']:.4f} | SSIM: {best_epoch['SSIM']:.4f}" 
                f"| bpa MSE: {best_epoch['bpa_MSE']:.4f} | bpa RMSE: {best_epoch['bpa_RMSE']:.4f} | bpa MAE: {best_epoch['bpa_MAE']:.4f}\n")

    for key in metrics.keys():
        metrics[key] += eval(key)

    # Save metrics to a JSON file
    with open(f"{model_folder}/metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics, model_path, optimizer_path, early_stopping, last_epoch


def to_float(x):
    return x.item() if torch.is_tensor(x) else float(x)

def early_stop(eval_mses, length=50, threshold=0.0):
    if len(eval_mses) < length + 1:
        return False
    recent_vals = eval_mses[-length:]

    xs = list(range(length))
    mean_x = sum(xs) / length
    mean_y = sum(recent_vals) / length

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, recent_vals))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return False

    slope = num / den
    return slope > threshold
