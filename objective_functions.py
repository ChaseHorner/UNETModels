import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure



class MSE(nn.Module):
    def forward(self, predictions, target, mask):
        masked_diff = (predictions - target) ** 2 * mask
        valid_count = mask.sum()
        return masked_diff.sum() / valid_count

class RMSE(nn.Module):
    def forward(self, predictions, target, mask):
        masked_diff = (predictions - target) ** 2 * mask
        valid_count = mask.sum()
        return torch.sqrt(masked_diff.sum() / valid_count)

class MAE(nn.Module):
    def forward(self, predictions, target, mask):
        masked_diff = torch.abs(predictions - target) * mask
        valid_count = mask.sum()
        return masked_diff.sum() / valid_count


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


        # Find bounding box over spatial dimensions (ignore channel dim)
        y, x = torch.nonzero(mask[0][0], as_tuple=True)  # take channel 0
        ymin, ymax = y.min(), y.max() + 1
        xmin, xmax = x.min(), x.max() + 1

        # Crop along H×W only
        cropped_pred = predictions[..., ymin:ymax, xmin:xmax]
        cropped_target = target[..., ymin:ymax, xmin:xmax]

        return self.ssim_metric(cropped_pred, cropped_target)

class SSIM_Loss(nn.Module):
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
    
class WholeFieldDiff(nn.Module):
    def forward(self, predictions, target, mask):
        diff = torch.sum(target * mask) - torch.sum(predictions * mask)
        return torch.abs(diff)
    
class BushelsPerAcreErrors(nn.Module):
    def forward(self, predictions, target, mask):
        count = torch.sum(mask).item()

        pred_avg = torch.sum(predictions * mask) / count
        target_avg = torch.sum(target * mask) / count

        MSE = (pred_avg - target_avg) ** 2
        MAE = torch.abs(pred_avg - target_avg)

        return MSE.item(), MAE.item()