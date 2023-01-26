from colossalai.amp import AMP_TYPE

DATA_DIR="/media/hdd/dhcp/dhcp_lores"
OUTPUT_DIR="/home/viscent/hdd/viscent/SR-UNet/experiments/output_vae_all_t1"
AUGMENTATION=True
INPUT_MODALITIES=["t1"]
OUTPUT_MODALITIES=["t1"]
BATCH_SIZE=3
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=300
N_SPLITS=5
F_MAPS=[16, 32, 64, 128, 256]

WARMUP_EPOCHS=50
LR=0.001
NUM_SAMPLES=None
LATENT_SIZE = F_MAPS[-1]
ALPHA = 0.000025
DOWN_FACTOR = 5




fp16=dict(
    mode=AMP_TYPE.TORCH
)
