from glob import glob

from matplotlib import image
from sklearn.utils import resample

from torch.utils.data import Dataset

import numpy as np
import os
import os.path as op
import glob
import sys
import logging
import SimpleITK as sitk
from tqdm import trange, tqdm
from tqdm.contrib.concurrent import thread_map
from multiprocessing import cpu_count
import torchio as tio
import h5py as h5
import torch
from torch import nn
from kornia import augmentation as K

class SynthDolphinDataset(Dataset):
    def __init__(self, data_dir, phase="train", num_samples=None, input_modalities=['t1'], output_modalities=['t1'], augmentation=False,down_factor=5):
        self.data_dir = data_dir
        self.phase = phase
        self.augmentation = augmentation
        
        self.input_dual_modal = True if len(input_modalities) == 2 else False
        self.output_dual_modal = True if len(output_modalities) == 2 else False
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        folder_name="preprocessed_h5"
        self.preprocessed_path = op.join(data_dir, folder_name)
        assert op.exists(self.preprocessed_path)
        self.list_basenames = sorted(glob.glob(op.join(self.preprocessed_path, '*.h5')))
        self.list_basenames = [op.basename(x).split('.')[0] for x in self.list_basenames]
    

        assert (num_samples is None) or num_samples >= 10, "num_samples must be >= 10"

        if num_samples is None:
            self.num_samples = len(self.list_basenames)
        else:
            self.num_samples = num_samples

        if phase == "train":
            self.list_basenames = self.list_basenames[:int(0.8 * self.num_samples)]
            self.num_samples = int(0.8 * self.num_samples)
        elif phase == "val":
            self.list_basenames = self.list_basenames[int(0.8 * self.num_samples):self.num_samples]
            self.num_samples = int(0.2 * self.num_samples)
        logging.info(f'Creating dataset with {self.num_samples} examples')
        logging.info(f'length of list_images_t1: {len(self.list_basenames)}')

        self.transform_spatial_1 = K.RandomRotation3D((15., 20., 20.), p=0.5,keepdim=True)
        self.transform_spatial_2 = K.RandomAffine3D((15., 20., 20.), p=0.4,keepdim=True)
        self.transform_intensity_1 = K.RandomMotionBlur3D(3, 35., 0.5, p=0.4,keepdim=True)
        self.transform_intensity_2 = lambda x: x + torch.randn_like(x)*15 if np.random.rand() < 0.4 else x

        self.images=[]
        self.gts=[]

        self.load()


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # with h5.File(self.preprocessed_path + f"/{self.list_basenames[idx]}.h5", 'r') as f:
        #     image = f['image'][:]
        #     gt = f['gt'][:]
        image = self.images[idx]
        gt = self.gts[idx]
        if self.augmentation:
            image = torch.from_numpy(image).unsqueeze(0)
            gt = torch.from_numpy(gt).unsqueeze(0)
            image = self.transform_spatial_1(image)
            image = self.transform_spatial_2(image)
            image = self.transform_intensity_1(image)
            image = self.transform_intensity_2(image)
            gt = self.transform_spatial_1(gt, params=self.transform_spatial_1._params)
            gt = self.transform_spatial_2(gt, params=self.transform_spatial_2._params)
            image = image.squeeze(0)
            gt = gt.squeeze(0)
            image = image.numpy()
            gt = gt.numpy()
        return image, gt

    def _load(self,x):
        basename = x
        assert op.exists(op.join(self.preprocessed_path, basename + '.h5'))
        with h5.File(op.join(self.preprocessed_path, basename + '.h5'), 'r') as f:
            image = []
            image.append(f['image_axi'])
            image.append(f['image_cor'])
            image.append(f['image_sag'])
            image = np.array(image)
            gt = []
            gt.append(f['image_gt'])
                
            gt = np.array(gt)
            image = image.astype(np.float32)
            gt = gt.astype(np.float32)
        return image, gt

    def load(self):
        # image_save_path = op.join(self.data_dir, 'images_{}_{}.npy'.format(self.input_modalities, self.output_modalities))

        # pool = mp.Pool(mp.cpu_count()-16)
        ret = thread_map(self._load,self.list_basenames, max_workers=1, total=self.num_samples)
        # ret = pool.imap_unordered(pmap, trange(num_samples))
        # pool.close()
        images,gts = zip(*ret)

        # for a in zip(list_images_t1, list_images_t2, list_labels):
        #     image, gt = _load(a)
        #     images.append(image)
        #     gts.append(gt)
        #     count += 1
        #     if count == num_samples:
        #         break
            
        self.images = np.array(images)
        self.gts = np.array(gts)
