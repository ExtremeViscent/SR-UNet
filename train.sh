#!/bin/bash

source ~/.bashrc
conda deactivate
conda activate bunet

torchrun train_vae.py --config=/media/hdd/viscent/SR-UNet/configs/config_vae_all_t2.py