from colossalai.amp import AMP_TYPE

DATA_DIR="/media/hdd/viscent/SynthSR/generated_data_multimodal"
BATCH_SIZE=4
IN_CHANNELS=2
OUT_CHANNELS=1
NUM_EPOCHS=200
F_MAPS=[16, 32, 64, 128, 256]

SMALL_DATA = False


fp16=dict(
    mode=AMP_TYPE.TORCH
)

# parallel = dict(tensor=dict(size=2, mode='1d'),)

