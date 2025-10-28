from pyexpat import model
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from config_loader import configs
from  matplotlib.colors import LinearSegmentedColormap


MAX = 150
cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def visualize_predictions(model, model_folder, model_path, dataset, num_images=10):
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

            inputs = {k: v.unsqueeze(0).to(device) for k, v in batch.items() if k != 'field_year'}

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
            diff_img = pred_img - target_img

            # Apply mask
            pred_masked = np.where(mask == 1.0, pred_img, np.nan)
            target_masked = np.where(mask == 1.0, target_img, np.nan)
            diff_masked = np.where(mask == 1.0, diff_img, np.nan)
            
            #Crop NaN borders for better visualization
            ys, xs = np.where(~np.isnan(target_masked))
            if ys.size > 0 and xs.size > 0:
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                pred_masked = pred_masked[y_min:y_max+1, x_min:x_max+1]
                target_masked = target_masked[y_min:y_max+1, x_min:x_max+1]
                diff_masked = diff_masked[y_min:y_max+1, x_min:x_max+1]

            plt.figure(figsize=(120, 40))
            display_list = [
                np.clip(pred_masked, 0, MAX),
                np.clip(target_masked, 0, MAX),
                np.clip(diff_masked, -MAX/2, MAX/2)
            ]
            titles = ['Prediction', 'Target', 'Difference (Pred - Target)']
            
            
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(titles[i], fontdict={'fontsize': 100}, pad=60)
                img = display_list[i]
                im = plt.imshow(img, cmap=cmap, vmin=-MAX*3/4 , vmax=MAX*3/4) if i == 2 else plt.imshow(img, cmap='Greens', vmin=0, vmax=MAX)
                plt.colorbar(im, fraction=0.05).ax.tick_params(labelsize=40)
                plt.axis('off')

            output_path = os.path.join(model_folder, f"{field_year}.png")
            plt.savefig(output_path)


if __name__ == "__main__":
    from data_pipeline.data_loader import FieldDataset
    import models.unet as unet

    dataset = FieldDataset(configs.DATASET_PATH).with_field_year()
    MODEL_NAME = 'UNET_v1.2.1'  # Change to the desired model variant
    MODEL_PATH = f'outputs/{MODEL_NAME}/{MODEL_NAME}_best.pt'

    unet_model = unet.Unet()
    visualize_predictions(unet_model, f'outputs/{MODEL_NAME}', MODEL_PATH, dataset, num_images=10)