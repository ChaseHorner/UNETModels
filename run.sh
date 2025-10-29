#!/bin/bash
#SBATCH --partition=sixhour
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c710h797@ku.edu
#SBATCH --ntasks=1
#SBATCH --mem=190G
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|q6000|q8000|l40|mi210"
#SBATCH --cpus-per-task=10

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=auto run.py "$1"
