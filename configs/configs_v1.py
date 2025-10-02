import datetime

# ====== Model =======
MODEL_NAME = 'UNET_2' # The name of the model, used for saving/loading
MODEL_FOLDER = './UNET_2' # The folder where model checkpoints and logs will be saved

BASE_MODEL = 'unet'  # 'unet' or 'ynet'
DATASET_PATH = '/resfs/GROUPS/KBS/kars_yield/prepped_data/training_tensors'

#Model description
''' 
Write here a short description of the model architecture,
any special layers, or anything else that might be useful to know
'''

#Model Info
C0 = 32
C1 = 64
C2 = 128
C3 = 256
C4 = 512
C5 = 1024
C6 = 2048
C7 = 4096

BATCH_SIZE = 1
ACCUMULATION_STEPS = 8
EPOCHS = 40

#Not implemented yet
CRITERION = 'L1'
OPTIMIZER = 'AdamW'
LEARNING_RATE = 1e-4
BETA1 = 0.5

TRAIN_VAL_SPLIT = 0.8


# ===== Inputs ======

INPUT_KEYS = ['lidar', 'sentinel']  # Options: 'lidar', 'sentinel', 'pre_season', 'in_season'

#Target Info
TARGET_SIZE = [1, 256, 256]

#Lidar Info
LIDAR_IN_CHANNELS = L1 = 5
LIDAR_SIZE = [L1, 2560, 2560]

#Sentinel Info
SEN_BANDS = 11
SEN_PERIODS = 21
S1 = SEN_BANDS * SEN_PERIODS + 1
SEN_SIZE = [S1, 256, 256]


#Field Info
field_var_guess = 50

#Number of channels after sentinel data is compressed
S1_guess = SEN_BANDS * SEN_PERIODS + field_var_guess


#Weather Info
pre_season_start_date = "10-01"
time_series_start_date = "03-01"
prediction_date = "09-01"

IN_SEASON_DAYS = (datetime.datetime.strptime(prediction_date, "%m-%d") - datetime.datetime.strptime(time_series_start_date, "%m-%d")).days
PRE_SEASON_DAYS = (
	datetime.datetime.strptime(time_series_start_date, "%m-%d").replace(year=2001) -
	datetime.datetime.strptime(pre_season_start_date, "%m-%d").replace(year=2000)
).days

WEATHER_IN_CHANNELS = 2

#This is the "rolling temporal window" the model uses to process weather data
IN_SEASON_KERNEL_SIZE = 3
PRE_SEASON_KERNEL_SIZE = 14

#Number of channels after weather data is compressed
W1 = 10
W2 = 1

BOTTLENECK_SIZE = [8, 8]