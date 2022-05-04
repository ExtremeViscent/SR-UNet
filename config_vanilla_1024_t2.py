from colossalai.amp import AMP_TYPE

DATA_DIR="/media/rosalyn/Storage/generated_data_T2"
OUTPUT_DIR="/home/rosalyn/Documents/dHCP/synthsr_data/trained_model_T2_BUNET"
BATCH_SIZE=3
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=800
F_MAPS=[16, 32, 64, 128, 256]

LR=0.001
SMALL_DATA = False
LATENT_SIZE = 1024
ALPHA = 0.00025



fp16=dict(
    mode=AMP_TYPE.TORCH
)

