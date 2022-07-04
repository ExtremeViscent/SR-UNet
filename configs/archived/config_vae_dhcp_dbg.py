from colossalai.amp import AMP_TYPE
import os

DATA_DIR="/media/hdd/dhcp/dhcp_lores"
OUTPUT_DIR="/media/hdd/viscent/SR-UNet/experiments/output_unet_dhcp_dbg"
AUGMENTATION=False
INPUT_MODALITIES=["t1"]
OUTPUT_MODALITIES=["t1"]
BATCH_SIZE=3
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=800
N_SPLITS=5
F_MAPS=[16, 32, 64, 128, 256]
VAE = True

WARMUP_EPOCHS=2
LR=0.001
NUM_SAMPLES=50
LATENT_SIZE = F_MAPS[-1]
ALPHA = 0.000025
DOWN_FACTOR = 5
DATASET='dHCP'

NNODES=1
NGPUS_PER_NODE=1
WORLD_SIZE=NNODES*NGPUS_PER_NODE

# fp16=dict(
#     mode=AMP_TYPE.TORCH
# )

# parallel = dict(
#     data=dict(size=WORLD_SIZE),
#     tensor=dict(mode='1d', size=WORLD_SIZE),
# )