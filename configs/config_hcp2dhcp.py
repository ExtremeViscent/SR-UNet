from colossalai.amp import AMP_TYPE
import os

DATA_DIR="/scratch/prj/bayunet/dhcp_lores"
OUTPUT_DIR="/scratch/prj/bayunet/experiments/output_hcp2dhcp"
CHECKPOINT="/scratch/prj/bayunet/experiments_0706/output_vae_hcp_t1_800_warmup/0/checkpoints/199.pth"
AUGMENTATION=False
INPUT_MODALITIES=["t1"]
OUTPUT_MODALITIES=["t1"]
BATCH_SIZE=6
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=1000
N_SPLITS=5
F_MAPS=[16, 32, 64, 128, 256]

WARMUP_EPOCHS=500
LR=0.001
NUM_SAMPLES=None
LATENT_SIZE = 1
ALPHA = 0.00025
DOWN_FACTOR = 5
DATASET='dHCP'