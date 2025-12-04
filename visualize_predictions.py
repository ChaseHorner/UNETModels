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

def visualize_predictions(model, model_folder, model_path, dataset, num_images=10, output_type='png', all_fields=False, hid=None):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Make sure to not sample more than available
    num_images = min(num_images, len(dataset)) if not all_fields else len(dataset)
    indices = torch.randperm(len(dataset))[:num_images] if not all_fields else torch.arange(len(dataset))

    if hid is not None:
        #Find index for specified hid
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample.get('hid', None) == hid:
                indices = torch.tensor([i])
                break

    if all_fields:
        bpa_diffs = []
    
    with torch.inference_mode():
         for idx in indices:
            sample = dataset[idx]

            field_year = sample.get('field_year', idx)
            hid = sample.get('hid', 'unknown')

            inputs = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if k != 'field_year' and k != 'hid'}

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

            field_diff = target_total - pred_total
            bpa_diff = target_bpa - pred_bpa

            if all_fields:
                bpa_diffs.append(bpa_diff)

            percent_diff = field_diff / target_total * 100
            
            #Calculate R²
            valid_mask = ~np.isnan(target_masked)
            if valid_mask.sum() > 0:
                y_true = target_masked[valid_mask]
                y_pred = pred_masked[valid_mask]
                y_mean = np.mean(y_true)
                ss_tot = np.sum((y_true - y_mean) ** 2)
                ss_res = np.sum((y_true - y_pred) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
            else:
                r2 = np.nan

            from sklearn.metrics import r2_score
            r2_sklearn = r2_score(y_true, y_pred) if valid_mask.sum() > 0 else np.nan
            
            print(f"Custom R²: {r2}, Sklearn R²: {r2_sklearn}")


            #Crop NaN borders for better visualization
            ys, xs = np.where(~np.isnan(target_masked))
            if ys.size > 0 and xs.size > 0:
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                pred_masked = pred_masked[y_min:y_max+1, x_min:x_max+1]
                target_masked = target_masked[y_min:y_max+1, x_min:x_max+1]
                diff_masked = diff_masked[y_min:y_max+1, x_min:x_max+1]

            # Create plots
            if output_type == 'png':
                display_list = [
                    np.clip(pred_masked, 0, MAX),
                    np.clip(target_masked, -MAX, MAX),
                    np.clip(diff_masked, -MAX/2, MAX/2)
                ]
                titles = ['Prediction', 'Target', 'Difference (Target-Pred)']
                subtitles = [f'Total Predicted: {pred_total:.2f} | BPA: {pred_bpa:.2f}',
                            f'Total Target: {target_total:.2f} | BPA: {target_bpa:.2f}',
                            f'Field Difference : {field_diff:.2f}\nBPA Difference: {bpa_diff:.2f}\nPercent Difference: {percent_diff:.2f}%\nR²: {r2:.4f}']
                
                plt.figure(figsize=(12, 6), constrained_layout=True)
                plt.suptitle(f'{field_year} | HID: {hid}', fontsize=12)
                for i in range(3):
                    plt.subplot(1, 3, i+1)
                    plt.title(titles[i], fontdict={'fontsize': 10}, pad=60)
                    plt.text(0.5, -0.18, subtitles[i], fontsize=8, ha='center', transform=plt.gca().transAxes)
                    img = display_list[i]
                    im = plt.imshow(img, cmap=cmap, vmin=-MAX*3/4 , vmax=MAX*3/4) if i == 2 else plt.imshow(img, cmap='Greens', vmin=0, vmax=MAX)
                    plt.colorbar(im, fraction=0.05).ax.tick_params(labelsize=8)
                    plt.axis('off')


                output_path = os.path.join(model_folder, f"{hid}.png")
                plt.savefig(output_path)
                plt.close()

            elif output_type == 'tiff':
                import rasterio
                from rasterio.transform import from_origin

                # Create tiffs folder
                tiffs_folder = os.path.join(model_folder, 'tiffs')
                os.makedirs(tiffs_folder, exist_ok=True)

                # Define transform (assuming pixel size of 1 and origin at (0,0); adjust as needed)
                transform = from_origin(0, 0, 1, 1)

                # Save prediction
                with rasterio.open(
                    os.path.join(tiffs_folder, f"{hid}_prediction.tiff"),
                    'w',
                    driver='GTiff',
                    height=pred_masked.shape[0],
                    width=pred_masked.shape[1],
                    count=1,
                    dtype=pred_masked.dtype,
                    crs='+proj=latlong',
                    transform=transform,
                ) as dst:
                    dst.write(pred_masked, 1)

                # Save difference
                with rasterio.open(
                    os.path.join(tiffs_folder, f"{hid}_difference.tiff"),
                    'w',
                    driver='GTiff',
                    height=diff_masked.shape[0],
                    width=diff_masked.shape[1],
                    count=1,
                    dtype=diff_masked.dtype,
                    crs='+proj=latlong',
                    transform=transform,
                ) as dst:
                    dst.write(diff_masked, 1)
    
    #Bell curve of BPA differences
    if all_fields:
        plt.figure(figsize=(20,10))
        plt.hist(bpa_diffs, bins='auto', color='blue')
        plt.title('Histogram of BPA Differences (Target - Prediction)', fontsize=20)
        plt.xlabel('BPA Difference', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        output_path = os.path.join(model_folder, f"losses_bell_curve.png")
        plt.savefig(output_path)
        plt.close()




if __name__ == "__main__":
    from data_pipeline.data_loader import FieldDataset
    import models.unet4 as unet

    dataset = FieldDataset(configs.DATASET_PATH, input_keys=configs.INPUT_KEYS, years=configs.VAL_YEARS).with_field_year_hid()
    MODEL_NAME = configs.MODEL_NAME  # Change to the desired model variant
    MODEL_PATH = f'{configs.MODEL_FOLDER}/{MODEL_NAME}_best_epoch292.pt'

    unet_model = unet.Unet4()
    visualize_predictions(unet_model, configs.MODEL_FOLDER, MODEL_PATH, dataset, num_images=10, output_type='png', all_fields=False, hid = 344)