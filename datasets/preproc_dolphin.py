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
import matplotlib.pyplot as plt

data_dir = '/home/levibaljer/FLYWHEEL_DOLPHIN/T2_Inputs_and_Targets/'
list_basenames = os.listdir(data_dir)
list_basenames = [op.join(data_dir,b) for b in list_basenames]
_list_basenames = list_basenames
_list_basenames = [op.basename(x) for x in list_basenames]

list_basenames.sort()

list_images_axi = [op.join(x, "brainex", "AXI", "brainex_inc.nii") for x in list_basenames]
list_images_cor = [op.join(x, "brainex", "COR", "brainex_inc.nii") for x in list_basenames]
list_images_sag = [op.join(x, "brainex", "SAG", "brainex_inc.nii") for x in list_basenames]

list_images_gt = [op.join(x, "brainex", "ABCD", "brainex_inc.nii") for x in list_basenames]

list_basenames = _list_basenames
print(list_basenames[0])
num_samples = len(list_basenames)

def _load(x):
    img_axi, img_cor, img_sag, img_gt, basename = x
    preprocessed_path = op.join(data_dir, "preprocessed_h5")
    subject = tio.Subject(
        image_axi = tio.ScalarImage(img_axi),
        image_cor = tio.ScalarImage(img_cor),
        image_sag = tio.ScalarImage(img_sag),
        image_gt = tio.ScalarImage(img_gt)
    )
    transform_1 = tio.Compose([
        tio.transforms.Resample((1.,1.,1.)),
    ])
    

    subject = transform_1(subject)
    edge_max = max(subject.image_axi.data.shape)
    padding = ((edge_max - subject.image_axi.data.shape[1]) // 2, 
                (edge_max - subject.image_axi.data.shape[2]) // 2,
                    (edge_max - subject.image_axi.data.shape[3]) // 2)
    transform_2 = tio.Compose([
        tio.Pad(padding),
        tio.transforms.Resize((160,160,160)),
        # tio.transforms.RandomNoise(0.5,(0,1)),
    ])

    subject = transform_2(subject)
    if not op.exists(preprocessed_path):
        os.makedirs(preprocessed_path)
    with h5.File(op.join(preprocessed_path, basename + '.h5'), 'w') as f:
        f.create_dataset('image_axi', data=subject.image_axi.data[0])
        f.create_dataset('image_cor', data=subject.image_cor.data[0])
        f.create_dataset('image_sag', data=subject.image_sag.data[0])
        f.create_dataset('image_gt', data=subject.image_gt.data[0])
        
def load(list_images_axi,list_images_cor, list_images_sag, list_images_gt, list_basenames):
    thread_map(_load, zip(list_images_axi, list_images_cor, list_images_sag, list_images_gt,list_basenames), max_workers=1, total=num_samples)

load(list_images_axi, list_images_cor, list_images_sag, list_images_gt,list_basenames)