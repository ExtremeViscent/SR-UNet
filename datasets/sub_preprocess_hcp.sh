#!/bin/bash
#SBATCH --output=/scratch/users/%u/preprocess_%j.out
#SBATCH --job-name=preprocess-hcp
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 24:00:00


source /users/k21113539/.bashrc
. "/users/k21113539/anaconda3/etc/profile.d/conda.sh"
conda activate cai

python /scratch/users/k21113539/SR-UNet/datasets/preprocess_hcp.py