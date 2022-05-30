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

data_dir = '/scratch/users/k21113539/HCP_1200'
list_dir = glob.glob(op.join(data_dir, '*'))
list_dir.sort()
list_basenames = [op.basename(x) for x in list_dir]
list_images_t1 = [op.join(x,'unprocessed','3T','T1w_MPR1',op.basename(x)+'_3T_T1w_MPR1.nii.gz') for x in list_dir]
list_images_t2 = [op.join(x,'unprocessed','3T','T2w_SPC1',op.basename(x)+'_3T_T2w_SPC1.nii.gz') for x in list_dir]

# list_dir = list_dir[:300]
# list_basenames = list_basenames[:300]
# list_images_t1 = list_images_t1[:300]
# list_images_t2 = list_images_t2[:300]

num_samples = len(list_dir)
spacing = [1.0,1.0,1.0]
spacing = np.array(spacing)
spacing *= 8
target_shape = (108, 145, 145)
factor = spacing[2] / spacing[0]
resize_transform = tio.Resize(target_shape=target_shape)
resample_transform = tio.Resample(target=spacing)
blur_transform = tio.RandomBlur(6)
transform  = tio.Compose([resample_transform,resize_transform,blur_transform])
transform_gt = resize_transform

def _load(x):
    img_t1, img_t2, basename = x
    preprocessed_path = op.join(data_dir, "preprocessed")
    image_t1 = sitk.ReadImage(img_t1)
    image_t2 = sitk.ReadImage(img_t2)
    if True:
        image_t1 = transform_gt(image_t1)
        image_t2 = transform_gt(image_t2)
    image_t1 = sitk.GetArrayFromImage(image_t1)
    image_t2 = sitk.GetArrayFromImage(image_t2)
    image_t1 = (image_t1 - np.mean(image_t1)) / np.std(image_t1)
    image_t2 = (image_t2 - np.mean(image_t2)) / np.std(image_t2)
    image_t1 = image_t1
    image_t2 = image_t2
    if not op.exists(preprocessed_path):
        os.makedirs(preprocessed_path)
    sitk.WriteImage(sitk.GetImageFromArray(image_t1), op.join(preprocessed_path, basename + '_t1.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(image_t2), op.join(preprocessed_path, basename + '_t2.nii.gz'))

def load(list_images_t1,list_images_t2,list_basenames):
    thread_map(_load, zip(list_images_t1, list_images_t2,list_basenames), max_workers=128, total=num_samples)

load(list_images_t1,list_images_t2,list_basenames)