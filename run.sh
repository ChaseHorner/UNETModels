#!/bin/bash
#SBATCH --partition=sixhour
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c710h797@ku.edu
#SBATCH --ntasks=1
#SBATCH --mem=190G
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu
#SBATCH --constraint="a40|l40|a100|q8000"

python run.py "$1"
