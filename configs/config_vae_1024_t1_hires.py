from colossalai.amp import AMP_TYPE

DATA_DIR="/media/hdd/dhcp/dhcp_hires"
OUTPUT_DIR="/home/viscent/hdd/viscent/SR-UNet/experiments/output_vae_1024_t1_hires"
AUGMENTATION=True
INPUT_MODALITIES=["t1"]
OUTPUT_MODALITIES=["t1"]
BATCH_SIZE=1
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=200
N_SPLITS=5
F_MAPS=[4, 8, 16, 32, 64]

LR=0.001
NUM_SAMPLES=300
LATENT_SIZE = F_MAPS[-1]
ALPHA = 0.00025
DOWN_FACTOR = 5




fp16=dict(
    mode=AMP_TYPE.TORCH
)
