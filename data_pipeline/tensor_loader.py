import os
import torch
import rasterio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


folder_path = "/resfs/GROUPS/KBS/kars_yield/prepped_data/training_dat_jk"
save_path = "/resfs/GROUPS/KBS/kars_yield/prepped_data/training_tensors"

TARGET_SIZE = [1, 256, 256]
LIDAR_SIZE = [5, 2560, 2560]
S2_SIZE = [231, 256, 256]  # 11 bands * 21 periods + 1 yield mask


def make_completed_tensors(save_path):
    completed_tensors = []
    for year in os.listdir(save_path):
        year_path = os.path.join(save_path, year)
        if not os.path.isdir(year_path):
            continue
        for field in os.listdir(year_path):
            field_path = os.path.join(year_path, field)
            if os.path.isdir(field_path):
                files = os.listdir(field_path)
                if all(f"{dt}.pt" in files for dt in ["lidar", "s2", "hrvst"]):
                    completed_tensors.append(f"{year}/{field}")
    with open(os.path.join(os.path.dirname(__file__), "completed_tensors.txt"), "w") as f:
        for item in completed_tensors:
            f.write(item + "\n")


def load_dataset(folder_path, save_path, completed_file="completed_tensors.txt", max_workers=8):
    completed_path = os.path.join(os.path.dirname(__file__), completed_file)
    if os.path.exists(completed_path):
        with open(completed_path, "r") as f:
            completed = {line.strip() for line in f}
    else:
        completed = set()

    tasks = []
    for year in os.listdir(folder_path):
        year_path = os.path.join(folder_path, year)
        if not os.path.isdir(year_path):
            continue

        for field in os.listdir(year_path):
            field_path = os.path.join(year_path, field)
            if not os.path.isdir(field_path):
                continue

            if f"{year}/{field}" in completed:
                continue

            output_path = os.path.join(save_path, year, field)
            tasks.append((field_path, output_path))

    print(f"🔄 Processing {len(tasks)} fields with {max_workers} workers...")

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(lambda args: load_field(*args), tasks), total=len(tasks)):
            pass



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

    shape_dict = {"lidar" : LIDAR_SIZE,
                    "s2" : S2_SIZE,
                    "hrvst" : TARGET_SIZE,
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
    load_dataset(folder_path, save_path, max_workers=24)
    # make_completed_tensors(save_path)