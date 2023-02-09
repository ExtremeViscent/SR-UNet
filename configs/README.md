# Configuration Files
This directory contains configuration files for the various purposes. We provide a sample configuration file at `configs/templates/config.py` which can be used as a template for your own configuration files.

The configuration files used for publication are as follows:
* `configs/CREATE/unet/config_unet_dhcp_t2_5000.py` - Configuration file training on the dHCP dataset.
* `configs/CREATE/unet/config_unet_hcp_t2_5000.py` - Configuration file training on the HCP dataset.
* `configs/CREATE/unet/config_unet_brats_t2_5000.py` - Configuration file training on the BraTS dataset.

## Datasets

The datasets and path used for publication are as follows:
* dHCP dataset, path: `/media/hdd/dhcp/dhcp_hires`
* HCP dataset, path: `/media/hdd/HCP_1200`
* BraTS dataset, path: `/media/hdd/BraTS2021/Training`

The parameters for the datasets can be found in the preprocessing files in `datasets/preproc_*.py`.