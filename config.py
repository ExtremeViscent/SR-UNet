from colossalai.amp import AMP_TYPE

DATA_DIR="/media/hdd/viscent/SynthSR/generated_data"
BATCH_SIZE=3
IN_CHANNELS=1
OUT_CHANNELS=1
NUM_EPOCHS=800
F_MAPS=[16, 32, 64, 128, 256]


fp16=dict(
    mode=AMP_TYPE.TORCH
)

# parallel = dict(tensor=dict(size=2, mode='1d'),)

