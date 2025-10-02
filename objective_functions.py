import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure



def weighted_l1_loss(predictions, target, mask, weight = 1.0):
    """
    Calculates the L1 loss with a per-element weight.
    This returns the SUM of weighted errors.
    """

    #TODO FIX THIS ONCE MASKS ARE FIXED
    weights = torch.where(mask == 0, weight, 0.0)

    l1_diff = torch.abs(predictions - target)
    
    # Apply weights and sum them up
    loss = (l1_diff * weights).sum()
    
    return loss

def weighted_PSNR(predictions, target, mask, weight = 1.0, data_range=200.0):
    """
    Calculates the Weighted Peak Signal-to-Noise Ratio (WPSNR).
    """
    #TODO FIX THIS ONCE MASKS ARE FIXED
    weights = torch.where(mask == 0, weight, 0.0)

    # Calculate squared error
    squared_error = (predictions - target) ** 2
    
    # Calculate weighted mean squared error (WMSE)
    sum_of_weights = torch.sum(weights)
    if sum_of_weights == 0:
        return torch.tensor(0.0)
        
    wmse = torch.sum(weights * squared_error) / sum_of_weights
    
    # If WMSE is zero, return infinity
    if wmse == 0:
        return torch.tensor(float('inf'))

    # Calculate WPSNR
    wpsnr_val = 10 * torch.log10((data_range ** 2) / wmse)
    
    return wpsnr_val


def cropped_SSIM(predictions, target, mask, data_range=200.0):
    """
    Calculates SSIM on the bounding box of the mask.
    """
    mask = mask.to(predictions.device)
    mask = torch.where(mask == 0, 1.0, 0.0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)

    # Find bounding box
    non_zero = torch.nonzero(mask)
    min_coords = non_zero.min(dim=0)[0]
    max_coords = non_zero.max(dim=0)[0] + 1
    slices = tuple(slice(min_c.item(), max_c.item()) for min_c, max_c in zip(min_coords, max_coords))

    cropped_pred = predictions[:, :, slices[0], slices[1]]
    cropped_target = target[:, :, slices[0], slices[1]]

    # Calculate SSIM
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(predictions.device)
    return ssim_metric(cropped_pred, cropped_target)
