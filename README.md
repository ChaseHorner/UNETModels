# UNETModels
A framework for training and evaluating UNet-based models on geospatial and remote sensing datasets. Designed to run efficiently on HPC GPU clusters with configurable pipelines and easy model scheduling.

## Installation
Install required packages for training:
```
pip install torch, torch.utils, torch.optim, torchmetrics, matplotlib
```

For data downloading and preprocessing:
```
pip install concurrent, tqdm, rasterio
```

## Getting Started
The main entry point for understanding and running the code is [Run.py](https://github.com/ChaseHorner/UNETModels/blob/main/run.py)


This framework is designed to kick off and monitor multiple training jobs on HPC GPUs via [scheduler.sh](https://github.com/ChaseHorner/UNETModels/blob/main/scheduler.sh) and [scheduler.py](https://github.com/ChaseHorner/UNETModels/blob/main/scheduler.py), but models can also be trained independently on a local device.

## Config Files
A fundamental aspect of the architecture is the use of configs.py files to mandate as much as possible about the model. An example of one of these config files is [configs.py](https://github.com/ChaseHorner/UNETModels/blob/main/configs.py)
The config defines every hyperparameter of the model, as well as inputs and outputs. It makes it possible to train many very different models at once.

In order to train a model, you must have a valid config file. The example is the default is none is set. 
Use the CONFIG environment variable to determine which config file to load

Use 
```
export CONFIG=path/to/configs.py
```
to set the environment variable before running the code.
This is done automatically when using the scheduler in line 32 of scheduler.py.
The magic takes place in [config_loader.py](https://github.com/ChaseHorner/UNETModels/blob/main/config_loader.py)


## Training a Model
To train an individual model on your device (instead of scheduling it) use [Run.py](https://github.com/ChaseHorner/UNETModels/blob/main/run.py)

Set up a model folder, named after the model.
Within the folder, put a configs.py and a status.json file with the correct model name and path to the folder. 
```
model_folder/
├──configs.py
├──status.json 
```
Set the CONFIG env variable to path/to/configs.py:
```
export CONFIG=path/to/configs.py
```
Run the training script:
```
python run.py path/to/model_folder
```

All outputs will be saved to that folder.


## Scheduling Multiple Models
*The scheduling process is specified for the KU CRC system, with tweaks, it should work for any HPC system.

Create the model folders and configs.py files for each model (No need to create a status.json file, scheduler.py does that for you)
Within [scheduler.py](https://github.com/ChaseHorner/UNETModels/blob/main/scheduler.py), update the model_paths list at the top of the file to your model paths

Run
```
sbatch scheduler.sh
```

