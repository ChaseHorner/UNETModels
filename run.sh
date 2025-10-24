#!/bin/bash
#SBATCH --partition=sixhour
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c710h797@ku.edu
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu

python run.py "$1"
