import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure



class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss that applies different weights to masked and unmasked regions.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, predictions, target, mask):
        weights = torch.where(mask == 1.0, self.weight, 1.0-self.weight)
        loss = (torch.abs(predictions - target) * weights).sum() / weights.sum()
        return loss

class WeightedL2Loss(nn.Module):
    """
    Weighted L2 Loss that applies different weights to masked and unmasked regions.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, predictions, target, mask):
        weights = torch.where(mask == 1.0, self.weight, 1.0-self.weight)
        loss = ((predictions - target) ** 2 * weights).sum() / weights.sum()
        return loss
    
class WeightedPSNR(nn.Module):
    """
    Weighted Peak Signal-to-Noise Ratio (WPSNR) that applies different weights to masked and unmasked regions.
    """
    def __init__(self, weight=1.0, data_range=350.0):
        super().__init__()
        self.weight = weight
        self.data_range = data_range

    def forward(self, predictions, target, mask):
        weights = torch.where(mask == 1.0, self.weight, 1.0-self.weight)

        squared_error = (predictions - target) ** 2
        sum_of_weights = torch.sum(weights)
        if sum_of_weights == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        wmse = torch.sum(weights * squared_error) / sum_of_weights
        wmse = torch.clamp(wmse, min=1e-12)

        wpsnr_val = 10 * torch.log10((self.data_range ** 2) / wmse)
        return wpsnr_val

class CroppedSSIM(nn.Module):
    """
    Computes SSIM only over the bounding box of a binary mask.
    """
    def __init__(self, data_range=350.0):
        super().__init__()
        self.data_range = data_range
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, predictions, target, mask):
        mask = mask.to(predictions.device)
        mask = (mask > 0).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)

        # Find bounding box over spatial dimensions (ignore channel dim)
        y, x = torch.nonzero(mask[0][0], as_tuple=True)  # take channel 0
        ymin, ymax = y.min(), y.max() + 1
        xmin, xmax = x.min(), x.max() + 1

        # Crop along H×W only
        cropped_pred = predictions[..., ymin:ymax, xmin:xmax]
        cropped_target = target[..., ymin:ymax, xmin:xmax]

        return self.ssim_metric(cropped_pred, cropped_target)

class cSSIM_Loss(nn.Module):
    """
    Loss based on Cropped SSIM.
    """
    def __init__(self, data_range=350.0):
        super().__init__()
        self.cropped_ssim = CroppedSSIM(data_range=data_range)

    def forward(self, predictions, target, mask):
        ssim_value = self.cropped_ssim(predictions, target, mask)
        loss = 1.0 - ssim_value
        return loss