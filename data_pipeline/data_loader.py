import os
import torch
from torch.utils.data import Dataset, DataLoader
import time

class FieldDataset(Dataset):
    def __init__(self, root_dir, input_keys=['lidar', 'sentinel', 'in_season', 'pre_season', 'auc','hmask'], years= None):
        '''
        root_dir: folder with subfolders per sample, each containing .pt files
        input_keys: list of keys for input tensors        '''
        self.root_dir = root_dir
        self.input_keys = input_keys

        self.samples = []
        
        for year in os.listdir(root_dir):
            if years is not None and year not in years:
                continue 

            year_path = os.path.join(root_dir, year)
            for field in os.listdir(year_path):
                field_path = os.path.join(year_path, field)
                if os.path.isdir(field_path):
                    self.samples.append(field_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        features = {

        'lidar' : torch.load(os.path.join(sample_dir, 'lidar.pt')),
        'sentinel' : torch.load(os.path.join(sample_dir, 's2.pt')),
        # 'weather_in_season' : torch.load(os.path.join(sample_dir, 'in_season.pt')),  
        # 'weather_pre_season' : torch.load(os.path.join(sample_dir, 'pre_season.pt')),

        'target' : torch.load(os.path.join(sample_dir, 'hrvst.pt')),
        'hmask' : torch.load(os.path.join(sample_dir, 'hmask.pt')),
        'auc' : torch.load(os.path.join(sample_dir, 'auc.pt')),
        'field_year' : sample_dir.split(os.sep)[-2] + '_' + sample_dir.split(os.sep)[-1]
        }

        returns = {key: features[key] for key in self.input_keys if key in features}
        returns['target'] = features['target']

        return returns
    
    def with_field_year(self):
        new_dataset = FieldDataset.__new__(FieldDataset)
        new_dataset.samples = self.samples 
        new_dataset.input_keys = self.input_keys + ['field_year']
        new_dataset.root_dir = self.root_dir
        return new_dataset

if __name__ == '__main__':
    start_time = time.time()
    dataset = FieldDataset(r'Z:\prepped_data\processed_tensors', input_keys=['lidar', 'sentinel'])
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    for lidar, sentinel, in_season, pre_season, target, auc in dataloader:
        print(f'Lidar batch shape: {lidar.shape}')
        print(f'Sentinel batch shape: {sentinel.shape}')
        print(f'In-season weather batch shape: {in_season.shape}')
        print(f'Pre-season weather batch shape: {pre_season.shape}')
        print(f'AUC batch shape: {auc.shape}')
        print(f'Target batch shape: {target.shape}')
        break  # Just test one batch
    end_time = time.time()
    print(f'Data loading time: {end_time - start_time:.2f} seconds')