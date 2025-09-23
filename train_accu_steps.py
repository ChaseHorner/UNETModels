import time
import torch
from torcheval.metrics.functional import peak_signal_noise_ratio
from torchmetrics.image import StructuralSimilarityIndexMeasure

import configs


def train_epoch(model, optimizer, criterion, train_dataloader, device):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    model.train()
    running_psnr, running_ssim, running_count = 0, 0, 0
    optimizer.zero_grad()

    for step, (lidar, sentinel, in_season, pre_season, labels) in enumerate(train_dataloader):
        lidar = lidar.to(device)
        sentinel = sentinel.to(device)
        in_season = in_season.to(device)
        pre_season = pre_season.to(device)
        labels = labels.to(device)

        predictions = model(lidar, sentinel, in_season, pre_season)

        # compute loss
        loss = criterion(predictions, labels)
        loss = loss / configs.ACCUMULATION_STEPS

        # backward
        loss.backward()

        # Gradient accumulation step
        if (step + 1) % configs.ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss = loss.item() * configs.ACCUMULATION_STEPS
        running_psnr += peak_signal_noise_ratio(predictions, labels).mean().item()
        running_ssim += ssim_metric(predictions, labels).mean().item()
        print(f"Training Step [{step+1}/{len(train_dataloader)}]", end = '\r')


        # Flush leftover gradients if dataset isn’t divisible by accumulation_steps
    if (step + 1) % configs.ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Average over number of batches
    num_batches = len(train_dataloader)
    return (
        running_psnr / num_batches,
        running_ssim / num_batches,
        running_loss / num_batches,
    )

def evaluate_epoch(model, criterion, valid_dataloader, device):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    model.eval()
    total_psnr, total_ssim, total_count = 0, 0, 0
    losses = []

    with torch.no_grad():
        for lidar, sentinel, in_season, pre_season, labels in valid_dataloader:
            lidar = lidar.to(device)
            sentinel = sentinel.to(device)
            in_season = in_season.to(device)
            pre_season = pre_season.to(device)
            labels = labels.to(device)

            predictions = model(lidar, sentinel, in_season, pre_season)

            loss = criterion(predictions, labels)
            losses.append(loss.item())


            total_psnr +=  peak_signal_noise_ratio(predictions, labels)
            total_ssim += ssim_metric(predictions, labels)
            total_count += 1

    epoch_psnr = total_psnr / total_count
    epoch_ssim = total_ssim / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_ssim, epoch_loss

def train_model(model, model_name, save_model, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device):
    train_psnrs, train_ssims, train_losses = [], [], []
    eval_psnrs, eval_ssims, eval_losses = [], [], []
    best_loss_eval = -1000
    times = []
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Training
        train_psnr, train_ssim, train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device)
        train_psnrs.append(to_float(train_psnr))
        train_ssims.append(to_float(train_ssim))
        train_losses.append(to_float(train_loss))

        # Evaluation
        eval_psnr, eval_ssim, eval_loss = evaluate_epoch(model, criterion, valid_dataloader, device)
        eval_psnrs.append(to_float(eval_psnr))
        eval_ssims.append(to_float(eval_ssim))
        eval_losses.append(to_float(eval_loss))

        # Save best model based on eval loss
        if best_loss_eval < eval_loss :
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')
            best_loss_eval = eval_loss
        times.append(time.time() - epoch_start_time)


        # Print loss, psnr end epoch
        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train psnr {:8.3f} | Train ssim {:8.3f} | Train Loss {:8.3f} "
            "| Valid psnr {:8.3f} | Valid ssim {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_psnr, train_ssim, train_loss, eval_psnr, eval_ssim, eval_loss
            )
        )
        print("-" * 59)

    # Load best model
    model.load_state_dict(torch.load(save_model + f'/{model_name}.pt'))
    model.eval()
    metrics = {
        'train_psnr': train_psnrs,
        'train_ssim': train_ssims,
        'train_loss': train_losses,
        'valid_psnr': eval_psnrs,
        'valid_ssim': eval_ssims,
        'valid_loss': eval_losses,
        'time': times
    }
    return model, metrics



def to_float(x):
    return x.item() if torch.is_tensor(x) else float(x)