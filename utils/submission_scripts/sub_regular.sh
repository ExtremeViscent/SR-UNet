#!/bin/bash
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_800.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_hcp_800.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_800_warmup.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_hcp_800_warmup.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_800_unet.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_hcp_800_unet.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_800_ls1.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_800_ls2.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_800_ls4.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_800_ls32.sh