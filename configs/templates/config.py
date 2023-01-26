from colossalai.amp import AMP_TYPE
import os

DATA_DIR = "/path/to/inputs/preprocessed_h5"
OUTPUT_DIR = "/path/to/outputs"
AUGMENTATION=False
INPUT_MODALITIES=["t2"]
OUTPUT_MODALITIES=["t2"]
BATCH_SIZE=6
IN_CHANNELS=1 # number of input channels. Set to 2 for T1 and T2
OUT_CHANNELS=1 # number of output channels. Set to 2 for T1 and T2
NUM_EPOCHS=4000 
N_SPLITS=5 # number of folds
F_MAPS=[16, 32, 64, 128, 256] # feature map sizes


LR=0.001 # learning rate
NUM_SAMPLES=None
DATASET='BraTS' # dHCP, HCP, BraTS are supported