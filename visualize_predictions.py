import torch
import matplotlib.pyplot as plt
import numpy as np


import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(66)


def visualize_predictions(model, model_folder, model_name, dataloader, num_images=3):
    model.load_state_dict(torch.load(model_folder + f'/{model_name}.pt'))
    model.to(device)
    model.eval()

    # Make sure to not sample more than available
    num_images = min(num_images, len(dataloader.dataset))
    indices = torch.randperm(len(dataloader.dataset))[:num_images]
    
    with torch.no_grad():
       for idx in indices:
            batch = dataloader.dataset[idx]
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}

            field_year = inputs.pop('field_year', idx)
            target = inputs.pop("target")
            
            predictions = model(**inputs)
    
            # move to CPU and convert to numpy
            target = target.cpu().numpy()
            predictions = predictions.cpu().numpy()
    
            plt.figure(figsize=(12, 4))
            display_list = [
                np.squeeze(predictions[0]), 
                np.squeeze(target[0]), 
                np.abs(np.squeeze(predictions[0]) - np.squeeze(target[0]))
                
            ]
            titles = ['Prediction', 'Target', 'Difference']
            
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(titles[i])
                img = display_list[i]
                plt.imshow(img)    
                plt.axis('off')

            output_path = os.path.join(model_folder, f"{field_year}.png")
            plt.savefig(output_path)
