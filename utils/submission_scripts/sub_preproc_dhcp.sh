#!/bin/bash
#SBATCH --output=/scratch/users/%u/SR-UNet/logs/preproc_dhcp_%j.out
#SBATCH --job-name=preproc-dhcp
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 40:00:00




source /users/k21113539/.bashrc

. "/users/k21113539/anaconda3/etc/profile.d/conda.sh"
conda activate cai



cd /scratch/users/k21113539/SR-UNet
python /scratch/users/k21113539/SR-UNet/datasets/preproc_dhcp.py