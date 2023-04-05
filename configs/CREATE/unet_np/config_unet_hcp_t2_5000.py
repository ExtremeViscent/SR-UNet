from colossalai.amp import AMP_TYPE
import os

DATA_DIR = "/scratch/prj/bayunet/HCP"
OUTPUT_DIR = "/scratch/prj/bayunet/experiments/hcp_t2_np"
AUGMENTATION=False
INPUT_MODALITIES=["t2"]
OUTPUT_MODALITIES=["t2"]
BATCH_SIZE=2
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=5000
N_SPLITS=5
F_MAPS=[64,64,64,64,64]

MODEL='unet3d'
apply_pooling=False

WARMUP_EPOCHS=None
LR=0.001
NUM_SAMPLES=None
LATENT_SIZE = 1
ALPHA = 0.00025
DOWN_FACTOR = 5
DATASET='HCP'
OPTMIZER = 'lamb'

NNODES=1
NGPUS_PER_NODE=2
WORLD_SIZE=NNODES*NGPUS_PER_NODE

fp16=dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    data=dict(size=WORLD_SIZE),
)