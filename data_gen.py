import torch
from torch.utils.data import Dataset, DataLoader
import configs
import model
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import os
import torch.optim as optim
from train import train_model

class RandomDataset(Dataset):
    def __init__(self, 
                 num_samples, 
                 lidar_input_shape = [configs.L1, configs._LIDAR_SIZE[0], configs._LIDAR_SIZE[1]],
                 sentinel_input_shape = [configs.S1, configs._SEN_SIZE[0], configs._SEN_SIZE[1]],
                 in_season_input_shape = [configs.WEATHER_IN_CHANNELS, configs.IN_SEASON_DAYS],
                 pre_season_input_shape = [configs.WEATHER_IN_CHANNELS, configs.PRE_SEASON_DAYS],
                 target_shape = [1, configs._SEN_SIZE[0], configs._SEN_SIZE[1]]):
        
        self.num_samples = num_samples
        self.lidar_input_shape = lidar_input_shape
        self.sentinel_input_shape = sentinel_input_shape
        self.in_season_input_shape = in_season_input_shape
        self.pre_season_input_shape = pre_season_input_shape
        self.target_shape = target_shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        a = torch.randn(self.lidar_input_shape)
        b = torch.randn(self.sentinel_input_shape)
        c = torch.randn(self.in_season_input_shape)
        d = torch.randn(self.pre_season_input_shape)
        e = torch.randn(self.target_shape)
        return a, b, c, d, e

def get_random_dataloader(num_samples=100, batch_size=configs.BATCH_SIZE, shuffle=True):
    dataset = RandomDataset(num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage:
if __name__ == "__main__":
    
    train_loader = get_random_dataloader(num_samples=200)
    val_loader = get_random_dataloader(num_samples=50, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_model = model.Unet()
    unet_model.to(device)  

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(unet_model.parameters(), lr=1e-4, betas=[0.5,0.999])
    EPOCHS = 10

    save_model = './UNET'
    os.makedirs(save_model, exist_ok = True)
    MODEL_NAME = 'unet_model_1'

    print(f"Training on {device} for {EPOCHS} epochs...")
    unet_model, metrics = train_model(
        unet_model, MODEL_NAME, save_model, optimizer, criterion, train_loader, val_loader, EPOCHS, device
    )

