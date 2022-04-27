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
    def __init__(self, data_dir, phase="train", num_samples=None, dual_modal=False):
        self.data_dir = data_dir
        self.phase = phase
        list_files = glob.glob(op.join(data_dir, '*.nii.gz'))
        self.ids = list(set([int(op.basename(f).split('_')[0]) for f in list_files]))
        # self.num_samples = len(self.ids)
        if num_samples is None:
            self.num_samples = len(self.ids)
        else:
            self.num_samples = max(num_samples, 10)
        if not self.ids:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        if phase == "train":
            self.ids = self.ids[:int(0.8 * self.num_samples)]
            self.num_samples = int(0.8 * self.num_samples)
        elif phase == "val":
            self.ids = self.ids[int(0.8 * self.num_samples):]
            self.num_samples = int(0.2 * self.num_samples)
        logging.info(f'Creating dataset with {self.num_samples} examples')
        if dual_modal:
            self.load_dual_modal(data_dir, phase)
        else:
            self.load(data_dir, phase)



    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        gt = self.gts[idx]
        # image = np.expand_dims(image, axis=0)
        # gt = np.expand_dims(gt, axis=0)
        return image, gt

    def load(self, data_dir, phase="train"):
        images = []
        gts = []
        for idx in trange(0, self.num_samples) if phase == "train" else trange(len(self.ids)-self.num_samples, len(self.ids)):
            image = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_image.nii.gz'))
            gt = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_target.nii.gz'))
            # label = sitk.ReadImage(op.join(data_dir, str(idx) + '_label.nii.gz'))
            images.append(sitk.GetArrayFromImage(image))
            gts.append(sitk.GetArrayFromImage(gt))
        images = np.array(images)
        gts = np.array(gts)
        self.images = images
        self.gts = gts

    def load_dual_modal(self, data_dir, phase="train"):
        images = []
        gts = []
        for idx in trange(0, self.num_samples) if phase == "train" else trange(len(self.ids)-self.num_samples, len(self.ids)):
            image_t1 = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_image_t1.nii.gz'))
            image_t2 = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_image_t2.nii.gz'))
            image = np.stack([sitk.GetArrayFromImage(image_t1), sitk.GetArrayFromImage(image_t2)], axis=0)
            gt = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_target.nii.gz'))
            # label = sitk.ReadImage(op.join(data_dir, str(idx) + '_label.nii.gz'))
            images.append(image)
            gts.append(sitk.GetArrayFromImage(gt))
        images = np.array(images)
        gts = np.array(gts)
        self.images = images
        self.gts = gts