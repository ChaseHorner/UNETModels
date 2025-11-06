from pyexpat import model
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from config_loader import configs
from  matplotlib.colors import LinearSegmentedColormap
import pandas as pd


MAX = 150
cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def visualize_predictions(model, model_folder, model_path, dataset, num_images=10, output_type='png'):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Make sure to not sample more than available
    num_images = min(num_images, len(dataset))
    indices = torch.randperm(len(dataset))[:num_images]

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
       for idx in indices:
            batch = dataloader.dataset[idx]
            # print(f"Visualizing sample {idx} from {batch.get('field_year', 'unknown')}")
            # print(batch['lidar'].shape, batch['sentinel'].shape, batch['hmask'].shape, batch['target'].shape)

            field_year = batch.get('field_year', idx)
            hid = batch.get('hid', 'unknown')

            inputs = {k: v.unsqueeze(0).to(device) for k, v in batch.items() if k != 'field_year' and k != 'hid'}

            target = inputs.pop("target")

            predictions = model(**inputs)
    
            # move to CPU and convert to numpy
            target = target.cpu().numpy()
            predictions = predictions.cpu().numpy()
            mask = inputs.get('hmask').cpu().numpy()

            # Only plot where mask == 1.0
            mask = np.squeeze(mask[0])  # shape: H x W
            pred_img = np.squeeze(predictions[0])
            target_img = np.squeeze(target[0])
            diff_img = target_img - pred_img

            # Apply mask
            pred_masked = np.where(mask == 1.0, pred_img, np.nan)
            target_masked = np.where(mask == 1.0, target_img, np.nan)
            diff_masked = np.where(mask == 1.0, diff_img, np.nan)
            
            #Get totals
            pred_total = np.nansum(pred_masked)
            pred_bpa = np.nanmean(pred_masked)

            target_total = np.nansum(target_masked)
            target_bpa = np.nanmean(target_masked)

            field_diff = pred_total - target_total

            percent_diff = field_diff / target_total * 100
            bpa_diff = (pred_bpa - target_bpa) / target_bpa * 100
            
            #Crop NaN borders for better visualization
            ys, xs = np.where(~np.isnan(target_masked))
            if ys.size > 0 and xs.size > 0:
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                pred_masked = pred_masked[y_min:y_max+1, x_min:x_max+1]
                target_masked = target_masked[y_min:y_max+1, x_min:x_max+1]
                diff_masked = diff_masked[y_min:y_max+1, x_min:x_max+1]

            display_list = [
                np.clip(pred_masked, 0, MAX),
                np.clip(target_masked, -MAX, MAX),
                np.clip(diff_masked, -MAX/2, MAX/2)
            ]
            titles = ['Prediction', 'Target', 'Difference (Target-Pred)']
            subtitles = [f'Total Predicted: {pred_total:.2f} | BPA: {pred_bpa:.2f}',
                         f'Total Target: {target_total:.2f} | BPA: {target_bpa:.2f}',
                         f'Field Difference : {field_diff:.2f}\nPercent Difference: {percent_diff:.2f}%']
            
            plt.figure(figsize=(120, 60), constrained_layout=True)
            plt.suptitle(f'{field_year} | HID: {hid}', fontsize=120)
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(titles[i], fontdict={'fontsize': 100}, pad=60)
                plt.text(0.5, -0.18, subtitles[i], fontsize=80, ha='center', transform=plt.gca().transAxes)
                img = display_list[i]
                im = plt.imshow(img, cmap=cmap, vmin=-MAX*3/4 , vmax=MAX*3/4) if i == 2 else plt.imshow(img, cmap='Greens', vmin=0, vmax=MAX)
                plt.colorbar(im, fraction=0.05).ax.tick_params(labelsize=40)
                plt.axis('off')


            output_path = os.path.join(model_folder, f"{hid}.{output_type}")
            plt.savefig(output_path)


if __name__ == "__main__":
    from data_pipeline.data_loader import FieldDataset
    import models.unet4 as unet

    dataset = FieldDataset(configs.DATASET_PATH, input_keys=configs.INPUT_KEYS, years=configs.VAL_YEARS).with_field_year_hid()
    MODEL_NAME = configs.MODEL_NAME  # Change to the desired model variant
    MODEL_PATH = f'{configs.MODEL_FOLDER}/{MODEL_NAME}_best_epoch195.pt'

    unet_model = unet.Unet4()
    visualize_predictions(unet_model, configs.MODEL_FOLDER, MODEL_PATH, dataset, num_images=1, output_type='png')