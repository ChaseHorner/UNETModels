#!/bin/bash

# List of model versions
models=("UNET_v1.2" "UNET_v1.3" "UNET_v1.4")

# Base path to configs
CONFIG_BASE="/kuhpc/scratch/kbs/c710h797/UNETModels/outputs"

# Loop through each model and submit sbatch
for model in "${models[@]}"; do
    CONFIG_PATH="${CONFIG_BASE}/${model}/configs.py"
    echo "Submitting job for $model with config $CONFIG_PATH"
    sbatch --export=CONFIG="$CONFIG" --job-name="$model" run.sh
done
