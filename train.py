import time
from flask import json
import torch
from config_loader import configs
from objective_functions import *
from torch.nn import L1Loss

def train_epoch(model, optimizer, criterion, train_dataloader, device, data_range=200.0, accu=False):
    model.train()
    running_l1, running_wL1, running_cSSIM, running_wPSNR = 0.0, 0.0, 0.0, 0.0

    if accu:
        optimizer.zero_grad()

    for step, batch in enumerate(train_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}

        target = inputs.pop("target")
        predictions = model(**inputs) 


        loss = criterion(predictions, target, inputs.get('hmask'))

        if accu:
            scaled_loss = loss / configs.ACCUMULATION_STEPS
            scaled_loss.backward()

            if (step + 1) % configs.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        running_l1 += L1Loss()(predictions, target).item()
        running_wL1 += WeightedL1Loss(weight=criterion.weight)(predictions, target, inputs.get('hmask')).item()
        running_wPSNR += WeightedPSNR(weight=criterion.weight)(predictions, target, inputs.get('hmask')).item()
        running_cSSIM += CroppedSSIM()(predictions, target, inputs.get('hmask')).item()

        print(f"Training Step [{step+1}/{len(train_dataloader)}]", end="\r")

    if accu and (step + 1) % configs.ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()


    num_batches = len(train_dataloader)
    return {
        "wPSNR": running_wPSNR / num_batches,
        "cSSIM": running_cSSIM / num_batches,
        "l1": running_l1 / num_batches,
        "wL1": running_wL1 / num_batches,
    }


def evaluate_epoch(model, criterion, valid_dataloader, device, data_range=200.0):
    model.eval()
    total_wPSNR, total_l1, total_cSSIM, total_wL1, total_count = 0, 0, 0, 0, 0
    losses = []

    with torch.no_grad():
        for batch in valid_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}

            target = inputs.pop("target")
            predictions = model(**inputs) 


            loss = criterion(predictions, target, inputs.get('hmask'))
            losses.append(loss.item())


            total_wL1 += WeightedL1Loss(weight=criterion.weight)(predictions, target, inputs.get('hmask')).item()
            total_wPSNR +=  WeightedPSNR(weight=criterion.weight)(predictions, target, inputs.get('hmask')).item()
            total_cSSIM += CroppedSSIM()(predictions, target, inputs.get('hmask')).item()
            total_l1 += L1Loss()(predictions, target).item()
            total_count += 1

    epoch_wPSNR = total_wPSNR / total_count
    epoch_cSSIM = total_cSSIM / total_count
    epoch_wl1 = total_wL1 / total_count
    epoch_l1 = total_l1 / total_count
    return {
        "wPSNR": epoch_wPSNR,
        "cSSIM": epoch_cSSIM,
        "l1": epoch_l1,
        "wL1": epoch_wl1
    }

def train_model(model, model_name, model_folder, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device, accu=False, data_range=200.0):
    train_psnrs, train_ssims, train_l1s, train_wlosses = [], [], [], []
    eval_psnrs, eval_ssims, eval_l1s, eval_wlosses = [], [], [], []
    best_wloss_eval = (float('inf'), -1)  # (loss, epoch)
    best_psnr_eval = (-float('inf'), -1)
    best_ssim_eval = (-float('inf'), -1)
    times = []
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Training
        train_metrics = train_epoch(model, optimizer, criterion, train_dataloader, device, accu=accu, data_range=data_range)
        train_psnrs.append(to_float(train_metrics["wPSNR"]))
        train_ssims.append(to_float(train_metrics["cSSIM"]))
        train_l1s.append(to_float(train_metrics["l1"]))
        train_wlosses.append(to_float(train_metrics["wL1"]))

        # Evaluation
        eval_metrics = evaluate_epoch(model, criterion, valid_dataloader, device, data_range=data_range)
        eval_psnrs.append(to_float(eval_metrics["wPSNR"]))
        eval_ssims.append(to_float(eval_metrics["cSSIM"]))
        eval_l1s.append(to_float(eval_metrics["l1"]))
        eval_wlosses.append(to_float(eval_metrics["wL1"]))

        # Save best model based on eval loss
        if best_wloss_eval[0] > eval_metrics["wL1"]:
            torch.save(model.state_dict(), model_folder + f'/{model_name}_lowest_wloss.pt')
            best_wloss_eval = (eval_metrics["wL1"], epoch)
        # Save best model based on eval psnr
        if best_psnr_eval[0] < eval_metrics["wPSNR"]:
            torch.save(model.state_dict(), model_folder + f'/{model_name}_highest_psnr.pt')
            best_psnr_eval = (eval_metrics["wPSNR"], epoch)
        # Save best model based on eval ssim
        if best_ssim_eval[0] < eval_metrics["cSSIM"]:
            torch.save(model.state_dict(), model_folder + f'/{model_name}_highest_ssim.pt')
            best_ssim_eval = (eval_metrics["cSSIM"], epoch)
        times.append(time.time() - epoch_start_time)


        # Print and log loss at end of epochs
        with open(f"{model_folder}/logs.txt", "a") as f:
            f.write("-" * 59 + "\n")
            f.write(
                "| End of epoch {:3d} | Time: {:5.2f}s | Train wPSNR {:8.3f} | Train cSSIM {:8.3f} | Train L1 {:8.3f} | Train wL1 {:8.3f} "
                "| Eval wPSNR {:8.3f} | Eval cSSIM {:8.3f} | Eval L1 {:8.3f} | Eval wL1 {:8.3f} ".format(
                    epoch, time.time() - epoch_start_time, train_metrics["wPSNR"], train_metrics["cSSIM"], train_metrics["l1"], train_metrics["wL1"],
                    eval_metrics["wPSNR"], eval_metrics["cSSIM"], eval_metrics["l1"], eval_metrics["wL1"]
                )
                + "\n"
            )
            f.write("-" * 59 + "\n")

        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train wPSNR {:8.3f} | Train cSSIM {:8.3f} | Train L1 {:8.3f} | Train wL1 {:8.3f} "
            "| Eval wPSNR {:8.3f} | Eval cSSIM {:8.3f} | Eval L1 {:8.3f} | Eval wL1 {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_metrics["wPSNR"], train_metrics["cSSIM"], train_metrics["l1"], train_metrics["wL1"],
                eval_metrics["wPSNR"], eval_metrics["cSSIM"], eval_metrics["l1"], eval_metrics["wL1"]
            )
        )
        print("-" * 59)
    
    # Save epoch number to a txt file
    with open(f"{model_folder}/{model_name}_saved_epochs.txt", "a") as f:
        f.write(f"Best wLoss Epoch: {best_wloss_eval[1]} with wLoss: {best_wloss_eval[0]:.4f}\n")
        f.write(f"Best PSNR Epoch: {best_psnr_eval[1]} with PSNR: {best_psnr_eval[0]:.4f}\n")
        f.write(f"Best SSIM Epoch: {best_ssim_eval[1]} with SSIM: {best_ssim_eval[0]:.4f}\n")

    metrics = {
        'train_psnr': train_psnrs,
        'train_ssim': train_ssims,
        'train_l1': train_l1s,
        'train_wloss': train_wlosses,
        'eval_psnr': eval_psnrs,
        'eval_ssim': eval_ssims,
        'eval_l1': eval_l1s,
        'eval_wloss': eval_wlosses,
        'time': times
    }

    # Save metrics to a JSON file
    with open(f"{model_folder}/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


def to_float(x):
    return x.item() if torch.is_tensor(x) else float(x)