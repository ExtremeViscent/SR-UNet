#!/bin/bash

# sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_t1_4000.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_t2_4000.sh
# sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_t1_4000_aug.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_t2_4000_aug.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_dhcp_t2_4000_pmse.sh

# sbatch /scratch/users/k21113539/SR-UNet/utils/sub_hcp_t1_4000.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_hcp_t2_4000.sh
# sbatch /scratch/users/k21113539/SR-UNet/utils/sub_hcp_t1_4000_aug.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_hcp_t2_4000_aug.sh
sbatch /scratch/users/k21113539/SR-UNet/utils/sub_hcp_t2_4000_pmse.sh