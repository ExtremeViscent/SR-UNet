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

data_dir = '/media/hdd/BraTS2020'
list_dir = glob.glob(op.join(data_dir, '*'))
list_dir.sort()
list_basenames = [op.basename(x) for x in list_dir]
list_images_t1 = [op.join(data_dir,x,x+'_t1.nii.gz') for x in list_basenames]
list_images_t2 = [op.join(data_dir,x,x+'_t2.nii.gz') for x in list_basenames]
list_masks = [op.join(data_dir,x,x+'_seg.nii.gz') for x in list_basenames]
num_samples = len(list_dir)
spacing = [1.5,1.5,5.0]
spacing = np.array(spacing)


def _load(x):
    img_t1, img_t2, mask, basename = x
    preprocessed_path = op.join(data_dir, "preprocessed_h5")
    subject = tio.Subject(
        image_t1 = tio.ScalarImage(img_t1),
        image_t2 = tio.ScalarImage(img_t2),
        mask = tio.LabelMap(mask)
    )
    transform_1 = tio.Compose([
        tio.transforms.RescaleIntensity((0., 1.)),
        tio.transforms.Resample(spacing),
        tio.transforms.ToCanonical(),
        tio.transforms.Resample((1.,1.,1.)),
        tio.transforms.RandomGamma((0.2,0.2))
    ])
    transform_1_gt = tio.Compose([
        tio.transforms.RescaleIntensity(0., 1.),
        tio.transforms.ToCanonical(),
        tio.transforms.Resample((1.,1.,1.)),
    ])
    subject_gt = transform_1_gt(subject)
    subject = transform_1(subject)
    edge_max = max(subject.image_t1.data.shape)
    padding = ((edge_max - subject.image_t1.data.shape[1]) // 2, 
                (edge_max - subject.image_t1.data.shape[2]) // 2,
                    (edge_max - subject.image_t1.data.shape[3]) // 2)
    transform_2 = tio.Compose([
        tio.Pad(padding),
        tio.transforms.Resize((160,160,160)),
    ])
    transform_2_gt = tio.Compose([
        tio.Pad(padding),
        tio.transforms.Resize((160,160,160)),
    ])
    subject_gt = transform_2_gt(subject_gt)
    subject = transform_2(subject)
    if not op.exists(preprocessed_path):
        os.makedirs(preprocessed_path)
    with h5.File(op.join(preprocessed_path, basename + '.h5'), 'w') as f:
        f.create_dataset('image_t1', data=subject.image_t1.data[0])
        f.create_dataset('image_t2', data=subject.image_t2.data[0])
        f.create_dataset('gt_t1', data=subject_gt.image_t1.data[0])
        f.create_dataset('gt_t2', data=subject_gt.image_t2.data[0])

def load(list_images_t1,list_images_t2,list_basenames):
    thread_map(_load, zip(list_images_t1, list_images_t2, list_masks,list_basenames), max_workers=1, total=num_samples)

load(list_images_t1,list_images_t2,list_basenames)