from colossalai.amp import AMP_TYPE
import os

DATA_DIR = "/media/hdd/dhcp/dhcp_hires"
OUTPUT_DIR = "/media/hdd/viscent/SR-UNet/experiments/output_cai_hcp_t2_dbg"
AUGMENTATION=False
INPUT_MODALITIES=["t2"]
OUTPUT_MODALITIES=["t2"]
BATCH_SIZE=1
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=1000
N_SPLITS=5
F_MAPS=[32, 64, 128, 256, 320]

WARMUP_EPOCHS=None
LR=0.001
NUM_SAMPLES=100
LATENT_SIZE = 128
ALPHA = 0.00025
DOWN_FACTOR = 5
DATASET='HCP'
OPTMIZER = 'lamb'

NNODES=1
NGPUS_PER_NODE=1
WORLD_SIZE=NNODES*NGPUS_PER_NODE

fp16=dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    data=dict(size=2),
)
