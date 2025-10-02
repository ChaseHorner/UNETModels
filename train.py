import time
import torch
import configs
from objective_functions import weighted_l1_loss, weighted_PSNR, cropped_SSIM

def train_epoch(model, optimizer, criterion, train_dataloader, device, data_range=200.0, accu=False):
    model.train()
    running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0

    if accu:
        optimizer.zero_grad()

    for step, batch in enumerate(train_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}

        target = inputs.pop("target")
        predictions = model(**inputs) 

        
        loss = weighted_l1_loss(predictions, target, inputs.get('hmask'), weight=1.0)

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


        running_loss += loss.item()
        running_psnr += weighted_PSNR(predictions, target, inputs.get('hmask'), data_range=data_range).item()
        running_ssim += cropped_SSIM(predictions, target, inputs.get('hmask'), data_range=data_range).item()

        print(f"Training Step [{step+1}/{len(train_dataloader)}]", end="\r")

    if accu and (step + 1) % configs.ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()


    num_batches = len(train_dataloader)
    return (
        running_psnr / num_batches,
        running_ssim / num_batches,
        running_loss / num_batches,
    )


def evaluate_epoch(model, criterion, valid_dataloader, device, data_range=200.0):
    model.eval()
    total_psnr, total_ssim, total_count = 0, 0, 0
    losses = []

    with torch.no_grad():
        for batch in valid_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}

            target = inputs.pop("target")
            predictions = model(**inputs) 


            loss = weighted_l1_loss(predictions, target, inputs.get('hmask'), weight=1.0)
            losses.append(loss.item())


            total_psnr +=  weighted_PSNR(predictions, target, inputs.get('hmask'), data_range=data_range).item()
            total_ssim += cropped_SSIM(predictions, target, inputs.get('hmask'), data_range=data_range).item()
            total_count += 1

    epoch_psnr = total_psnr / total_count
    epoch_ssim = total_ssim / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_ssim, epoch_loss

def train_model(model, model_name, model_folder, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device, accu=False, data_range=200.0):
    train_psnrs, train_ssims, train_losses = [], [], []
    eval_psnrs, eval_ssims, eval_losses = [], [], []
    best_loss_eval = (float('inf'), -1)  # (loss, epoch)
    best_psnr_eval = (-float('inf'), -1)
    best_ssim_eval = (-float('inf'), -1)
    times = []
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Training
        train_psnr, train_ssim, train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device, accu=accu, data_range=data_range)
        train_psnrs.append(to_float(train_psnr))
        train_ssims.append(to_float(train_ssim))
        train_losses.append(to_float(train_loss))

        # Evaluation
        eval_psnr, eval_ssim, eval_loss = evaluate_epoch(model, criterion, valid_dataloader, device, data_range=data_range)
        eval_psnrs.append(to_float(eval_psnr))
        eval_ssims.append(to_float(eval_ssim))
        eval_losses.append(to_float(eval_loss))

        # Save best model based on eval loss
        if best_loss_eval[0] > eval_loss :
            torch.save(model.state_dict(), model_folder + f'/{model_name}_lowest_loss.pt')
            best_loss_eval = (eval_loss, epoch)
        # Save best model based on eval psnr
        if best_psnr_eval[0] < eval_psnr:
            torch.save(model.state_dict(), model_folder + f'/{model_name}_highest_psnr.pt')
            best_psnr_eval = (eval_psnr, epoch)
        # Save best model based on eval ssim
        if best_ssim_eval[0] < eval_ssim:
            torch.save(model.state_dict(), model_folder + f'/{model_name}_highest_ssim.pt')
            best_ssim_eval = (eval_ssim, epoch)
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
    
    # Save epoch number to a txt file

    with open(f"{model_folder}/{model_name}_saved_epochs.txt", "a") as f:
        f.write(f"Best Loss Epoch: {best_loss_eval[1]} with Loss: {best_loss_eval[0]:.4f}\n")
        f.write(f"Best PSNR Epoch: {best_psnr_eval[1]} with PSNR: {best_psnr_eval[0]:.4f}\n")
        f.write(f"Best SSIM Epoch: {best_ssim_eval[1]} with SSIM: {best_ssim_eval[0]:.4f}\n")

    # Load best model
    # model.load_state_dict(torch.load(model_folder + f'/{model_name}.pt'))
    # model.eval()
    metrics = {
        'train_psnr': train_psnrs,
        'train_ssim': train_ssims,
        'train_loss': train_losses,
        'valid_psnr': eval_psnrs,
        'valid_ssim': eval_ssims,
        'valid_loss': eval_losses,
        'time': times
    }
    return metrics


def to_float(x):
    return x.item() if torch.is_tensor(x) else float(x)