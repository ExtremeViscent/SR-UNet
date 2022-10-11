from colossalai.amp import AMP_TYPE
import os

DATA_DIR="/media/hdd/dhcp/dhcp_lores"
OUTPUT_DIR="/media/hdd/viscent/SR-UNet/experiments/output_vae_dhcp_t1_800_warmup"
# Preds, checkpoints, and logs will be saved here
AUGMENTATION=False
# If True, use data augmentation
INPUT_MODALITIES=["t1"]
# Three modalities are supported: t2 axial, sagittal, and coronal
OUTPUT_MODALITIES=["t1"]
BATCH_SIZE=6
IN_CHANNELS=1
# Number of input channels (changed on change of input modality)
OUT_CHANNELS=1
NUM_EPOCHS=1000
N_SPLITS=5
F_MAPS=[16, 32, 64, 128, 256]

# WARMUP_EPOCHS=500
# Deprecated
LR=0.001
# Initial learning rate, using cosine annealing
NUM_SAMPLES=None
# For debugging, set to a small number like 100
# LATENT_SIZE = 1
# Deprecated
ALPHA = 0.00025
# Weight of the KL divergence loss/Sinkhorn loss
# Enabled only if VAE=True
DATASET='dHCP'

# Defined in ./datasets

VAE=False
# If True, use divergence loss during training

DIV_LOSS = 'sinkhorn'
# 'sinkhorn' or 'kl'
RECON_LOSS = 'mse'
# 'mse' or 'ssim' (Structural Similarity Index, super slow, not working, not recommended)


# NNODES=1
# NGPUS_PER_NODE=4
# WORLD_SIZE=NNODES*NGPUS_PER_NODE

# fp16=dict(
#     mode=AMP_TYPE.TORCH
# )

# parallel = dict(
#     data=dict(size=WORLD_SIZE),
#     tensor=dict(mode='1d', size=WORLD_SIZE),
# )