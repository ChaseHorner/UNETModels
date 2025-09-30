import os
import torch
import rasterio
import numpy as np
from .. import configs


folder_path = r"z:\prepped_data\training_dat_jk"
save_path = r"Z:\prepped_data\training_tensors"



def load_dataset(folder_path, save_path):
    for year in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, year)):
            continue

        year_path = os.path.join(folder_path, year)
        for field in os.listdir(year_path):
            field_path = os.path.join(year_path, field)
            if os.path.isdir(field_path):
                output_path = os.path.join(save_path, year, field)
                load_field(field_path, output_path)


def load_field(field_path, output_path, dtype=torch.float32):
    lidar_tensors = []
    s2_tensors = []
    hrvst_tensor = None
    yield_mask = None

    for data_type in os.listdir(field_path):
        data_type_path = os.path.join(field_path, data_type)
        if not os.path.isdir(data_type_path):
            continue

        data = sorted(os.listdir(data_type_path))  # ensure consistent order
        for file in data:
            file_path = os.path.join(data_type_path, file)

            if "hmsk" in file and file.endswith('.tif'):
                with rasterio.open(file_path) as src:
                        arr = src.read().astype(np.float32)
                        if arr.ndim == 2:
                            arr = arr[None, :, :]  # add channel dimension
                        yield_mask = torch.from_numpy(arr).type(dtype)

            elif file.endswith('.tif'):
                with rasterio.open(file_path) as src:
                    arr = src.read().astype(np.float32)
                    if arr.ndim == 2:
                        arr = arr[None, :, :]  # add channel dimension
                    tensor = torch.from_numpy(arr).type(dtype)
                
                    if data_type == 'lidar':
                        lidar_tensors.append(tensor)
                    elif data_type == 's2':
                        s2_tensors.append(tensor)
                    elif data_type == 'hrvst':
                        hrvst_tensor = tensor
                    else:
                        print(f"Unknown data type {data_type} in {field_path}, skipping {file_path}")

    lidar_tensor = torch.cat(lidar_tensors, dim=0)
    s2_tensor = torch.cat(s2_tensors + [yield_mask], dim=0)

    shape_dict = {"lidar" : configs.LIDAR_SIZE,
                    "s2" : configs.SEN_SIZE,
                    "hrvst" : configs.TARGET_SIZE,
                    }
    
    for data_type, final_tensor in zip(['lidar', 's2', 'hrvst'], [lidar_tensor, s2_tensor, hrvst_tensor]):
        if data_type in shape_dict:
            expected_shape = shape_dict[data_type]
            if list(final_tensor.shape) != expected_shape:
                raise ValueError(f"Shape mismatch for {data_type} in {field_path}. Expected {expected_shape}, got {list(final_tensor.shape)}")
        
        # Save tensor
        os.makedirs(output_path, exist_ok=True)
        save_file = os.path.join(output_path, f"{data_type}.pt")
        torch.save(final_tensor, save_file)
        print(f"Saved {save_file}, shape {final_tensor.shape}")


if __name__ == "__main__":
    load_dataset(folder_path, save_path)