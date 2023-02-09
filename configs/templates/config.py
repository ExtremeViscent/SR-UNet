from colossalai.amp import AMP_TYPE
import os

DATA_DIR = "/path/to/inputs/preprocessed_h5"
OUTPUT_DIR = "/path/to/outputs"
AUGMENTATION=False  # data augmentation
INPUT_MODALITIES=["t2"] # input modalities. Set to ["t1", "t2"] for T1 and T2
OUTPUT_MODALITIES=["t2"]  # output modalities. Set to ["t1", "t2"] for T1 and T2
BATCH_SIZE=6  # batch size
IN_CHANNELS=1 # number of input channels. Set to 2 for T1 and T2
OUT_CHANNELS=1 # number of output channels. Set to 2 for T1 and T2
NUM_EPOCHS=4000   # number of epochs
N_SPLITS=5 # number of folds for cross validation
F_MAPS=[16, 32, 64, 128, 256] # feature map sizes
OPTIMIZER='adam' # optimizer, adam or lamb


LR=0.001 # learning rate
NUM_SAMPLES=None  # Dataset size. Set to None for full dataset
DATASET='BraTS' # dHCP, HCP, BraTS are supported