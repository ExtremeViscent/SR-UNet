from colossalai.amp import AMP_TYPE
import os

DATA_DIR = "/scratch/prj/bayunet/HCP_1200"
OUTPUT_DIR = "/scratch/prj/bayunet/experiments/output_vae_hcp_t1_ls1_noaug"
AUGMENTATION=False
INPUT_MODALITIES=["t1"]
OUTPUT_MODALITIES=["t1"]
BATCH_SIZE=6
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=1000
N_SPLITS=5
F_MAPS=[16, 32, 64, 128, 256]

WARMUP_EPOCHS=None
LR=0.001
NUM_SAMPLES=None
LATENT_SIZE = 1
ALPHA = 0.00025
DOWN_FACTOR = 5
DATASET='HCP'
AUGMENTATION=False
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