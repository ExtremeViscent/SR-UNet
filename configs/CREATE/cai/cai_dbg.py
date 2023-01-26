from colossalai.amp import AMP_TYPE
import os

DATA_DIR = "/scratch/prj/bayunet/dhcp_lores"
OUTPUT_DIR = "/scratch/prj/bayunet/experiments/unet/dhcp_t2"
AUGMENTATION=False
INPUT_MODALITIES=["t2"]
OUTPUT_MODALITIES=["t2"]
BATCH_SIZE=6
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=4000
N_SPLITS=5
F_MAPS=[16, 32, 64, 128, 256]

VAE=False

WARMUP_EPOCHS=None
LR=0.001
NUM_SAMPLES=None
LATENT_SIZE = 1
ALPHA = 0.00025
DOWN_FACTOR = 5
DATASET='dHCP'

fp16=dict(
    mode=AMP_TYPE.TORCH
)

parallel = dict(
    data=dict(size=WORLD_SIZE),
    tensor=dict(mode='1d', size=WORLD_SIZE),
)