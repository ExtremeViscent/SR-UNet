from glob import glob

from matplotlib import image

from torch.utils.data import Dataset

import numpy as np
import os
import os.path as op
import glob
import sys
import logging
import SimpleITK as sitk
from tqdm import trange


class SynthdHCPDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        list_files = glob.glob(op.join(data_dir, '*.nii.gz'))
        self.ids = list(set([int(op.basename(f).split('_')[0]) for f in list_files]))
        # self.num_samples = len(self.ids)
        self.num_samples = max(self.ids) + 1
        if not self.ids:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {self.num_samples} examples')
        self.load(data_dir)



    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        gt = self.gts[idx]
        image = np.expand_dims(image, axis=0)
        gt = np.expand_dims(gt, axis=0)
        return image, gt

    def load(self, data_dir):
        images = []
        gts = []
        for idx in trange(0, self.num_samples):
            image = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_image.nii.gz'))
            gt = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_target.nii.gz'))
            # label = sitk.ReadImage(op.join(data_dir, str(idx) + '_label.nii.gz'))
            images.append(sitk.GetArrayFromImage(image))
            gts.append(sitk.GetArrayFromImage(gt))
        images = np.array(images)
        gts = np.array(gts)
        self.images = images
        self.gts = gts
