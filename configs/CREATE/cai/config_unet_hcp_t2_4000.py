from colossalai.amp import AMP_TYPE
import os

DATA_DIR = "/scratch/prj/bayunet/HCP_1200"
OUTPUT_DIR = "/scratch/prj/bayunet/experiments/unet/hcp_t2_cai"
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
DATASET='HCP'

NNODES=1
NGPUS_PER_NODE=2
WORLD_SIZE=NNODES*NGPUS_PER_NODE

fp16=dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    data=dict(size=WORLD_SIZE),
)
