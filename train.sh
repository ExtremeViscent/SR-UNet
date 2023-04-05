#!/bin/bash

source ~/.bashrc
conda deactivate
conda activate bunet    # activate the environments

torchrun train.py --config=configs/CREATE/unet/config_unet_brats_t2_5000.py