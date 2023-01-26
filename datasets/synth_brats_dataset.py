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

class SynthBraTSDataset(Dataset):
    def __init__(self, data_dir, phase="train", num_samples=None, input_modalities=['t1'], output_modalities=['t1'], augmentation=False,down_factor=5):
        self.data_dir = data_dir
        self.phase = phase
        self.augmentation = augmentation
        self.down_factor = down_factor
        self.input_dual_modal = True if len(input_modalities) == 2 else False
        self.output_dual_modal = True if len(output_modalities) == 2 else False
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        folder_name="preprocessed_h5"
        self.list_dir = glob.glob(op.join(data_dir,folder_name, '*.h5'))
        self.list_dir.sort()
        self.list_basenames = [op.basename(x).split('.')[0] for x in self.list_dir]
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

        self.images=[]
        self.gts=[]

        self.load()



    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        gt = self.gts[idx]
        return image, gt





    def _load(self,basename):
        # print(op.join(self.preprocessed_path, basename + '.h5'))
        if op.exists(op.join(self.preprocessed_path, basename + '.h5')):
            with h5.File(op.join(self.preprocessed_path, basename + '.h5'), 'r') as f:
                image = []
                for modality in self.input_modalities:
                    image.append(f['image_'+modality][:])
                image = np.array(image)
                gt = []
                for modality in self.output_modalities:
                    gt.append(f['gt_'+modality][:])
                gt = np.array(gt)
                image = image.astype(np.float32)
                gt = gt.astype(np.float32)
            return image, gt
        else:
            return None, None
        

    def load(self):
        ret = thread_map(self._load, self.list_basenames, max_workers=1, total=self.num_samples)

        images,gts = zip(*ret)

        self.images = np.array(images)
        self.gts = np.array(gts)

