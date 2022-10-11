from colossalai.amp import AMP_TYPE
import os

DATA_DIR = "/media/hdd/HCP_1200"
OUTPUT_DIR = "/home/levibaljer/SR-UNet/experiments/output_vae_hcp_t2_4000"
AUGMENTATION=False
INPUT_MODALITIES=["t2"]
OUTPUT_MODALITIES=["t2"]
BATCH_SIZE=2
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=1000
N_SPLITS=5
F_MAPS=[16, 32, 64, 128, 256]

WARMUP_EPOCHS=None
LR=0.001
NUM_SAMPLES=100
LATENT_SIZE = 128
ALPHA = 0.00025
DOWN_FACTOR = 5
DATASET='HCP'

DIV_LOSS = 'sinkhorn'
RECON_LOSS = 'patch_mse'
# NNODES=1
# NGPUS_PER_NODE=4
# WORLD_SIZE=NNODES*NGPUS_PER_NODE

# fp16=dict(
#     mode=AMP_TYPE.TORCH,
# )

# parallel = dict(
#     data=dict(size=WORLD_SIZE),
#     tensor=dict(mode='1d', size=WORLD_SIZE),
# )