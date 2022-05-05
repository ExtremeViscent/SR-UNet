from colossalai.amp import AMP_TYPE

DATA_DIR="/media/hdd/viscent/SynthSR/generated_data_multimodal"
OUTPUT_DIR="/home/viscent/hdd/viscent/SR-UNet/output_vae_1024_t1"
BATCH_SIZE=3
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=200
F_MAPS=[16, 32, 64, 128, 256]

LR=0.001
SMALL_DATA = False
LATENT_SIZE = 1024
ALPHA = 0.00025



fp16=dict(
    mode=AMP_TYPE.TORCH
)

