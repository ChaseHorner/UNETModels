import torch
import matplotlib.pyplot as plt
import numpy as np


import os

import configs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def visualize_predictions(model, model_folder, model_name, dataset, num_images=3):
    model.load_state_dict(torch.load(model_folder + f'/{model_name}.pt'))
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
    
            plt.figure(figsize=(120, 40))
            display_list = [
                np.squeeze(predictions[0]), 
                np.squeeze(target[0]), 
                np.abs(np.squeeze(predictions[0]) - np.squeeze(target[0]))
                
            ]
            titles = ['Prediction', 'Target', 'Difference']
            
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(titles[i], fontdict={'fontsize': 60})
                img = display_list[i]
                plt.imshow(img)    
                plt.axis('off')

            output_path = os.path.join(model_folder, f"{field_year}.png")
            plt.savefig(output_path)


if __name__ == "__main__":
    from data_pipeline.data_loader import FieldDataset
    import models.unet as unet

    dataset = FieldDataset(configs.DATASET_PATH).with_field_year()
    model_folder = './outputs/UNET_v0.2'
    MODEL_NAME = 'UNET_v0.2_highest_psnr'  # Change to the desired model variant

    unet_model = unet.Unet()
    visualize_predictions(unet_model, model_folder, MODEL_NAME, dataset, num_images=2)