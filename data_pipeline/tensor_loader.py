import os
import torch
import rasterio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

'''
This module provides functions to load geospatial datasets from a specified folder structure, process them into tensors, and save them in the resFS'''


folder_path = "/resfs/GROUPS/KBS/kars_yield/prepped_data/training_dat_drylnd_jk" #David's folder
save_path = "/resfs/GROUPS/KBS/kars_yield/prepped_data/training_tensors_v2"

TARGET_SIZE = [1, 256, 256]
LIDAR_SIZE = [5, 2560, 2560]
S2_SIZE = [231, 256, 256]  # 11 bands * 21 periods
S2_REDUCED_SIZE = [126, 256, 256]  # 6 bands * 21 periods
AUC_SIZE = [1, 256, 256] 


def make_completed_tensors(save_path):
    '''Generates a list of completed tensors and saves it to a text file.'''
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

def check_incomplete_tensors(folder_path, save_path):
    '''Checks for incomplete tensors by comparing the existing tensors against the completed list.'''
    make_completed_tensors(save_path)
    completed_path = os.path.join(os.path.dirname(__file__), "completed_tensors.txt")
    with open(completed_path, "r") as f:
        completed = {line.strip() for line in f}

    # Check for incomplete tensors
    for year in os.listdir(folder_path):
        year_path = os.path.join(folder_path, year)
        if not os.path.isdir(year_path):
            continue
        for field in os.listdir(year_path):
            field_path = os.path.join(year_path, field)
            if os.path.isdir(field_path):
                if f"{year}/{field}" not in completed:
                    print(f"Incomplete tensor found: {year}/{field}")


def load_dataset(folder_path, save_path, completed_file="completed_tensors.txt", max_workers=8):
    '''Loads the dataset from the specified folder path, processes each field into tensors, and saves them to the save path.'''

    #Checking for already completed tensors
    completed_path = os.path.join(os.path.dirname(__file__), completed_file)
    if os.path.exists(completed_path):
        with open(completed_path, "r") as f:
            completed = {line.strip() for line in f}
    else:
        completed = set()

    #Creating list of tasks to process in parallel
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

    print(f"Processing {len(tasks)} fields with {max_workers} workers...")

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(_load_field_wrapper, tasks), total=len(tasks)):
            pass

def _load_field_wrapper(args):
    return load_field(*args)

def load_field(field_path, output_path, dtype=torch.float32):
    '''Loads and processes a single field's data from the specified path and saves the tensors to the output path.'''
    lidar_tensors = []
    s2_tensors = []
    s2_reduced_tensors = []

    for data_type in os.listdir(field_path):
        data_type_path = os.path.join(field_path, data_type)
        if not os.path.isdir(data_type_path):
            continue

        data = sorted(os.listdir(data_type_path))  # ensure consistent order
        for file in data:
            file_path = os.path.join(data_type_path, file)

            if "NDVI_AuC_p150_300" in file and file.endswith('.tif'):
                with rasterio.open(file_path) as src:
                        arr = src.read().astype(np.float32)
                        if arr.ndim == 2:
                            arr = arr[None, :, :]  # add channel dimension
                        auc_tensor = torch.from_numpy(arr).type(dtype)

            elif "hmsk" in file and file.endswith('.tif'):
                with rasterio.open(file_path) as src:
                        arr = src.read().astype(np.float32)
                        if arr.ndim == 2:
                            arr = arr[None, :, :]  # add channel dimension
                        hmask_tensor = torch.from_numpy(arr).type(dtype)

            elif "hrvst" in file and file.endswith('.tif'):
                hid = file.split('_hid_')[1].split('_')[0].split('.')[0] # extract hid from filename
                with rasterio.open(file_path) as src:
                        arr = src.read().astype(np.float32)
                        if arr.ndim == 2:
                            arr = arr[None, :, :]  # add channel dimension
                        hrvst_tensor = torch.from_numpy(arr).type(dtype)

            elif file.endswith('.tif') and data_type == 's2':
                with rasterio.open(file_path) as src:
                    arr = src.read().astype(np.float32)
                    if arr.ndim == 2:
                        arr = arr[None, :, :]
                    tensor = torch.from_numpy(arr).type(dtype)
                    s2_tensors.append(tensor)

                    #Get the bands 4,5,6,8A,11,NDVI for reduced
                    arr = arr[[2, 4, 5, 7, 8, 10], :, :]
                    tensor_reduced = torch.from_numpy(arr).type(dtype)
                    s2_reduced_tensors.append(tensor_reduced)

            elif file.endswith('.tif') and data_type == 'lidar':
                with rasterio.open(file_path) as src:
                    arr = src.read().astype(np.float32)
                    if arr.ndim == 2:
                        arr = arr[None, :, :]  # add channel dimension
                    tensor = torch.from_numpy(arr).type(dtype)
                    lidar_tensors.append(tensor)

    # Concatenate tensors with multiple channels along the batch dimension
    s2_tensor = torch.cat(s2_tensors, dim=0)
    lidar_tensor = torch.cat(lidar_tensors, dim=0)
    s2_reduced_tensor = torch.cat(s2_reduced_tensors, dim=0)

    shape_dict = {"lidar" : LIDAR_SIZE,
                    "s2" : S2_SIZE,
                    "s2_reduced" : S2_REDUCED_SIZE,
                    "hmask" : TARGET_SIZE,
                    "hrvst" : TARGET_SIZE,
                    "auc_150_300_only" : AUC_SIZE
                    }
    
    for data_type, final_tensor in zip(['lidar', 's2', 's2_reduced', 'hmask', 'hrvst', 'auc_150_300_only'], [lidar_tensor, s2_tensor, s2_reduced_tensor, hmask_tensor, hrvst_tensor, auc_tensor]):
        
        # Validate tensor shape
        if data_type in shape_dict:
            expected_shape = shape_dict[data_type]
            if list(final_tensor.shape) != expected_shape:
                raise ValueError(f"Shape mismatch for {data_type} in {field_path}. Expected {expected_shape}, got {list(final_tensor.shape)}")
        
        # Save tensor
        os.makedirs(output_path, exist_ok=True)
        save_file = os.path.join(output_path, f"{data_type}.pt")
        torch.save(final_tensor, save_file)
        print(f"Saved {save_file}, shape {final_tensor.shape}", end="\r")

    # Save hid to a text file
    with open(os.path.join(output_path, f"hid.txt"), "w") as f:
        f.write(hid)



if __name__ == "__main__":
    load_dataset(folder_path, save_path, max_workers=96)
    # make_completed_tensors(save_path)
    # check_incomplete_tensors(folder_path, save_path)