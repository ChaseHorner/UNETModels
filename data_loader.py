import os
import torch
from torch.utils.data import Dataset, DataLoader
import configs
import time

class FieldDataset(Dataset):
    def __init__(self, root_dir, input_keys=['lidar', 'sentinel', 'in_season', 'pre_season'], target_key='target'):
        """
        root_dir: folder with subfolders per sample, each containing .pt files
        input_keys: list of keys for input tensors
        target_key: key for target tensor
        """
        self.root_dir = root_dir
        self.input_keys = input_keys
        self.target_key = target_key

        self.samples = []
        for year in os.listdir(root_dir):
            year_path = os.path.join(root_dir, year)
            for field in os.listdir(year_path):
                field_path = os.path.join(year_path, field)
                if os.path.isdir(field_path):
                    self.samples.append(field_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        lidar = torch.load(os.path.join(sample_dir, 'lidar.pt'))
        sentinel = torch.load(os.path.join(sample_dir, 'sentinel.pt'))
        # in_season = torch.load(os.path.join(sample_dir, 'in_season.pt'))
        # pre_season = torch.load(os.path.join(sample_dir, 'pre_season.pt'))
        in_season = torch.zeros([configs.WEATHER_IN_CHANNELS, configs.IN_SEASON_DAYS])
        pre_season = torch.zeros([configs.WEATHER_IN_CHANNELS, configs.PRE_SEASON_DAYS])
        target = torch.load(os.path.join(sample_dir, 'target.pt'))

        return lidar, sentinel, in_season, pre_season, target


start_time = time.time()
dataset = FieldDataset(r"Z:\prepped_data\processed_tensors", input_keys=['lidar', 'sentinel'])
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

for lidar, sentinel, in_season, pre_season, target in dataloader:
    print(f"Lidar batch shape: {lidar.shape}")
    print(f"Sentinel batch shape: {sentinel.shape}")
    print(f"In-season weather batch shape: {in_season.shape}")
    print(f"Pre-season weather batch shape: {pre_season.shape}")
    print(f"Target batch shape: {target.shape}")
    break  # Just test one batch
end_time = time.time()
print(f"Data loading time: {end_time - start_time:.2f} seconds")
