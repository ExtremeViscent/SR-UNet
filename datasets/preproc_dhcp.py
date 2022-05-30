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

data_dir = '/media/hdd/dhcp/dhcp_lores'
list_basenames = glob.glob(op.join(data_dir,'labels', '*.nii.gz'))
list_basenames.sort()
print(len(list_basenames))
list_images_t1 = [x.replace('_desc-drawem9_dseg.nii.gz', '_T1w_brain.nii.gz') for x in list_basenames]
list_images_t2 = [x.replace('_desc-drawem9_dseg.nii.gz', '_T2w_brain.nii.gz') for x in list_basenames]
list_images_t1 = [x.replace('_desc-drawem9_dseg_1mm.nii.gz', '_T1w_brain_1mm.nii.gz') for x in list_images_t1]
list_images_t2 = [x.replace('_desc-drawem9_dseg_1mm.nii.gz', '_T2w_brain_1mm.nii.gz') for x in list_images_t2]
list_images_t1 = [x.replace(op.join(data_dir, 'labels'), op.join(data_dir, 'images_t1')) for x in list_images_t1]
list_images_t2 = [x.replace(op.join(data_dir, 'labels'), op.join(data_dir, 'images_t2')) for x in list_images_t2]
list_basenames = [op.basename(x).split('_')[0] for x in list_images_t1]
# list_dir = list_dir[:300]
# list_basenames = list_basenames[:300]
# list_images_t1 = list_images_t1[:300]
# list_images_t2 = list_images_t2[:300]

num_samples = len(list_basenames)
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
    gt_t1 = image_t1
    gt_t2 = image_t2
    if True:
        image_t1 = transform(image_t1)
        image_t2 = transform(image_t2)
        gt_t1 = transform_gt(gt_t1)
        gt_t2 = transform_gt(gt_t2)
    image_t1 = sitk.GetArrayFromImage(image_t1)
    image_t2 = sitk.GetArrayFromImage(image_t2)
    gt_t1 = sitk.GetArrayFromImage(gt_t1)
    gt_t2 = sitk.GetArrayFromImage(gt_t2)
    image_t1 = (image_t1 - np.mean(image_t1)) / np.std(image_t1)
    image_t2 = (image_t2 - np.mean(image_t2)) / np.std(image_t2)
    gt_t1 = (gt_t1 - np.mean(gt_t1)) / np.std(gt_t1)
    gt_t2 = (gt_t2 - np.mean(gt_t2)) / np.std(gt_t2)
    if not op.exists(preprocessed_path):
        os.makedirs(preprocessed_path)
    sitk.WriteImage(sitk.GetImageFromArray(image_t1), op.join(preprocessed_path, basename + '_t1.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(image_t2), op.join(preprocessed_path, basename + '_t2.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(gt_t1), op.join(preprocessed_path, basename + '_gt_t1.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(gt_t2), op.join(preprocessed_path, basename + '_gt_t2.nii.gz'))

def _load_h5(x):
    img_t1, img_t2, basename = x
    preprocessed_path = op.join(data_dir, "preprocessed_h5")
    image_t1 = sitk.ReadImage(img_t1)
    image_t2 = sitk.ReadImage(img_t2)
    gt_t1 = image_t1
    gt_t2 = image_t2
    if True:
        image_t1 = transform(image_t1)
        image_t2 = transform(image_t2)
        gt_t1 = transform_gt(gt_t1)
        gt_t2 = transform_gt(gt_t2)
    image_t1 = sitk.GetArrayFromImage(image_t1)
    image_t2 = sitk.GetArrayFromImage(image_t2)
    gt_t1 = sitk.GetArrayFromImage(gt_t1)
    gt_t2 = sitk.GetArrayFromImage(gt_t2)
    image_t1 = (image_t1 - np.mean(image_t1)) / np.std(image_t1)
    image_t2 = (image_t2 - np.mean(image_t2)) / np.std(image_t2)
    gt_t1 = (gt_t1 - np.mean(gt_t1)) / np.std(gt_t1)
    gt_t2 = (gt_t2 - np.mean(gt_t2)) / np.std(gt_t2)
    if not op.exists(preprocessed_path):
        os.makedirs(preprocessed_path)
    with h5.File(op.join(preprocessed_path, basename + '.h5'), 'w') as f:
        f.create_dataset('image_t1', data=image_t1)
        f.create_dataset('image_t2', data=image_t2)
        f.create_dataset('gt_t1', data=gt_t1)
        f.create_dataset('gt_t2', data=gt_t2)

def load(list_images_t1,list_images_t2,list_basenames):
    thread_map(_load_h5, zip(list_images_t1, list_images_t2,list_basenames), max_workers=1, total=num_samples)

load(list_images_t1,list_images_t2,list_basenames)