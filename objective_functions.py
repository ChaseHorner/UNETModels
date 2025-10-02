import torch


def weightedL1Loss(predictions, target, weight):
    # Make sure mask is float
    weights = (target != 0).float() * weight + (target == 0).float() * (1 - weight)
    l1 = (torch.abs(predictions - target) * weights).sum() / weights.sum().clamp(min=1.0)
    return l1

def weightedMSELoss(predictions, target, weight):
    # Make sure mask is float
    weights = (target != 0).float() * weight + (target == 0).float() * (1 - weight)
    mse = ((predictions - target) ** 2 * weights).sum() / weights.sum().clamp(min=1.0)
    return mse

def combinedLoss(predictions, target, weight, alpha=0.5):
    l1 = weightedL1Loss(predictions, target, weight)
    mse = weightedMSELoss(predictions, target, weight)
    return alpha * l1 + (1 - alpha) * mse

def weightedPSNRLoss(predictions, target, weight, data_range=200.0):
    weights = (target != 0).float() * weight + (target == 0).float() * (1 - weight)
    mse = ((predictions - target) ** 2 * weights).sum() / weights.sum().clamp(min=1.0)
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return -psnr