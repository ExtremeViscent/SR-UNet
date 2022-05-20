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

class SynthHCPDataset(Dataset):
    def __init__(self, data_dir, phase="train", num_samples=None, input_modalities=['t1'], output_modalities=['t1'], augmentation=False,down_factor=5):
        self.data_dir = data_dir
        self.phase = phase
        self.augmentation = augmentation
        self.down_factor = down_factor
        self.input_dual_modal = True if len(input_modalities) == 2 else False
        self.output_dual_modal = True if len(output_modalities) == 2 else False
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities

        self.list_dir = glob.glob(op.join(data_dir,'preprocessed', '*.h5'))
        self.list_dir.sort()
        self.list_basenames = [op.basename(x) for x in self.list_dir]



        folder_name="preprocessed"
        self.preprocessed_path = op.join(data_dir, folder_name)

        assert (num_samples is None) or num_samples >= 10, "num_samples must be >= 10"

        if num_samples is None:
            self.num_samples = len(self.list_dir)
        else:
            self.num_samples = num_samples
        if not self.list_dir:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        if phase == "train":
            self.list_dir = self.list_dir[:int(0.8 * self.num_samples)]
            self.list_basenames = self.list_basenames[:int(0.8 * self.num_samples)]
            self.num_samples = int(0.8 * self.num_samples)
        elif phase == "val":
            self.list_dir = self.list_dir[int(0.8 * self.num_samples):self.num_samples]
            self.list_basenames = self.list_basenames[int(0.8 * self.num_samples):self.num_samples]
            self.num_samples = int(0.2 * self.num_samples)
        logging.info(f'Creating dataset with {self.num_samples} examples')

        spacing = [1.0,1.0,1.0]
        spacing = np.array(spacing)
        spacing *= down_factor
        target_shape = (320, 320, 256)
        factor = spacing[2] / spacing[0]
        aniso_transform = tio.RandomAnisotropy(axes=2, downsampling=factor)
        resize_transform = tio.Resize(target_shape=target_shape)
        resample_transform = tio.Resample(target=spacing)
        self.transform  = tio.Compose([resample_transform,resize_transform])
        self.transform_gt = resize_transform

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
        return image, gt





    def _load(self,basename):
        with h5.File(op.join(self.data_dir,'preprocessed', basename), 'r') as f:
            if self.input_dual_modal:
                image = np.stack(f['image_t1'][:], f['image_t2'][:])
            else:
                image = np.expand_dims(f['image_{}'.format(self.input_modalities[0])][:], axis=0)
            if self.output_dual_modal:
                gt = np.stack(f['gt_t1'][:], f['gt_t2'][:])
            else:
                gt = np.expand_dims(f['gt_{}'.format(self.output_modalities[0])][:], axis=0)
            image = image.astype(np.float32)
            gt = gt.astype(np.float32)
        return image, gt
        # image_t1 = sitk.ReadImage(img_t1)
        # image_t2 = sitk.ReadImage(img_t2)
        # gt_t1 = image_t1
        # gt_t2 = image_t2
        # if self.augmentation:
        #     image_t1 = self.transform(image_t1)
        #     image_t2 = self.transform(image_t2)
        #     gt_t1 = self.transform_gt(gt_t1)
        #     gt_t2 = self.transform_gt(gt_t2)
        # image_t1 = sitk.GetArrayFromImage(image_t1)
        # image_t2 = sitk.GetArrayFromImage(image_t2)
        # gt_t1 = sitk.GetArrayFromImage(gt_t1)
        # gt_t2 = sitk.GetArrayFromImage(gt_t2)
        # image = np.stack([image_t1, image_t2], axis=0)
        # gt = np.stack([gt_t1, gt_t2], axis=0)
        # image = image.astype(np.float32)
        # gt = gt.astype(np.float32)
        # with h5.File(op.join(self.preprocessed_path, basename + '.h5'), 'w') as f:
        #     f.create_dataset('image', data=image)
        #     f.create_dataset('gt', data=gt)
        # return image, gt
        

    def load(self):
        ret = thread_map(self._load,self.list_basenames, max_workers=32, total=self.num_samples)
        images,gts = zip(*ret)
        self.images = np.array(images)
        self.gts = np.array(gts)
