#!/bin/bash
#SBATCH --partition=kbs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c710h797@ku.edu
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err

python -u  scheduler.py 
