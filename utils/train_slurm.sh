#!/bin/bash
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --job-name=bunet
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t 24:00:00


export MASTER_PORT=11451
export WORLD_SIZE=8
export NPROC_PER_NODE=4
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export NNODES=$SLURM_JOB_NUM_NODES
export NODE_RANK=$SLURM_NODEID
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /users/k21113539/.bashrc

. "/users/k21113539/anaconda3/etc/profile.d/conda.sh"

conda activate cai

nvidia-smi

cd /scratch/users/k21113539/SR-UNet
# python /scratch/users/k21113539/SR-UNet/train_vae_bottleneck.py --config /scratch/users/k21113539/SR-UNet/configs/CREATE/config_vae_bn_t1.py
torchrun --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK --nproc_per_node $NPROC_PER_NODE /scratch/users/k21113539/SR-UNet/train_vae_bottleneck.py --config /scratch/users/k21113539/SR-UNet/configs/CREATE/config_vae_dhcp_t1.py
# srun -p gpu -N 2 --gres=gpu:4 --ntasks-per-node=4 python /scratch/users/k21113539/SR-UNet/train_vae_bottleneck.py --config /scratch/users/k21113539/SR-UNet/configs/CREATE/config_vae_bn_t1.py