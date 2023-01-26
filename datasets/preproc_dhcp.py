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

data_dir = '/media/hdd/dhcp/dhcp_hires'
list_basenames = glob.glob(op.join(data_dir,'labels', '*.nii.gz'))
list_basenames.sort()
print(len(list_basenames))
list_images_t1 = [x.replace('_desc-drawem9_dseg.nii.gz', '_T1w_brain.nii.gz') for x in list_basenames]
list_images_t2 = [x.replace('_desc-drawem9_dseg.nii.gz', '_T2w_brain.nii.gz') for x in list_basenames]
list_images_t1 = [x.replace('_desc-drawem9_dseg_1mm.nii.gz', '_T1w_brain_1mm.nii.gz') for x in list_images_t1]
list_images_t2 = [x.replace('_desc-drawem9_dseg_1mm.nii.gz', '_T2w_brain_1mm.nii.gz') for x in list_images_t2]
# TODO: delete this
list_images_t1 = [x.replace(op.join(data_dir, 'labels'), op.join(data_dir, 'images_t1')) for x in list_images_t1]
list_images_t2 = [x.replace(op.join(data_dir, 'labels'), op.join(data_dir, 'images_t2')) for x in list_images_t2]
list_basenames = [op.basename(x).split('_')[0] for x in list_images_t1]
landmarks_t1 = op.join(data_dir, 'landmarks_t1.npy')
landmarks_t2 = op.join(data_dir, 'landmarks_t2.npy')
landmarks_dict = {'image_t1':landmarks_t1,'image_t2':landmarks_t2}
# list_dir = list_dir[:300]
# list_basenames = list_basenames[:300]
# list_images_t1 = list_images_t1[:300]
# list_images_t2 = list_images_t2[:300]

num_samples = len(list_basenames)
# Low resolution of fake Hyperfine
spacing = [5.,1.5,1.5]
spacing = np.array(spacing)
# transform = tio.Compose([
#     tio.transforms.RescaleIntensity(0., 1.),
#     tio.transforms.ToCanonical(),
#     tio.transforms.RandomBlur((2,2)),
#     tio.transforms.RandomMotion(degrees=5.,translation=2.,num_transforms=20),
#     tio.transforms.RandomNoise(1.5,(0,1)),
#     tio.transforms.Resample(spacing),
#     tio.transforms.Resample((1.,1.,1.)),
#     tio.transforms.CropOrPad((128,144,144)),
# ])
# transform_gt = tio.Compose([
#     tio.transforms.RescaleIntensity(0., 1.),
#     tio.transforms.ToCanonical(),
#     tio.transforms.Resample(spacing),
#     tio.transforms.Resample((1.,1.,1.)),
#     tio.transforms.CropOrPad((128,144,144)),
# ])

# def _load(x):
#     img_t1, img_t2, basename = x
#     preprocessed_path = op.join(data_dir, "preprocessed")
#     image_t1 = sitk.ReadImage(img_t1)
#     image_t2 = sitk.ReadImage(img_t2)
#     gt_t1 = image_t1
#     gt_t2 = image_t2
#     if True:
#         image_t1 = transform(image_t1)
#         image_t2 = transform(image_t2)
#         gt_t1 = transform_gt(gt_t1)
#         gt_t2 = transform_gt(gt_t2)
#     image_t1 = sitk.GetArrayFromImage(image_t1)
#     image_t2 = sitk.GetArrayFromImage(image_t2)
#     gt_t1 = sitk.GetArrayFromImage(gt_t1)
#     gt_t2 = sitk.GetArrayFromImage(gt_t2)
#     image_t1 = (image_t1 - np.mean(image_t1)) / np.std(image_t1)
#     image_t2 = (image_t2 - np.mean(image_t2)) / np.std(image_t2)
#     gt_t1 = (gt_t1 - np.mean(gt_t1)) / np.std(gt_t1)
#     gt_t2 = (gt_t2 - np.mean(gt_t2)) / np.std(gt_t2)
#     if not op.exists(preprocessed_path):
#         os.makedirs(preprocessed_path)
#     sitk.WriteImage(sitk.GetImageFromArray(image_t1), op.join(preprocessed_path, basename + '_t1.nii.gz'))
#     sitk.WriteImage(sitk.GetImageFromArray(image_t2), op.join(preprocessed_path, basename + '_t2.nii.gz'))
#     sitk.WriteImage(sitk.GetImageFromArray(gt_t1), op.join(preprocessed_path, basename + '_gt_t1.nii.gz'))
#     sitk.WriteImage(sitk.GetImageFromArray(gt_t2), op.join(preprocessed_path, basename + '_gt_t2.nii.gz'))

# def _load_h5(x):
#     img_t1, img_t2, basename = x
#     preprocessed_path = op.join(data_dir, "preprocessed_h5_d5b3")
#     image_t1 = sitk.ReadImage(img_t1)
#     image_t2 = sitk.ReadImage(img_t2)
#     gt_t1 = image_t1
#     gt_t2 = image_t2
#     if True:
#         image_t1 = transform(image_t1)
#         image_t2 = transform(image_t2)
#         gt_t1 = transform_gt(gt_t1)
#         gt_t2 = transform_gt(gt_t2)
#     image_t1 = sitk.GetArrayFromImage(image_t1)
#     image_t2 = sitk.GetArrayFromImage(image_t2)
#     gt_t1 = sitk.GetArrayFromImage(gt_t1)
#     gt_t2 = sitk.GetArrayFromImage(gt_t2)
#     image_t1 = (image_t1 - np.mean(image_t1)) / np.std(image_t1)
#     image_t2 = (itio.

def get_bbox(img, lb):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r>lb)[0][[0,-1]]
    cmin, cmax = np.where(c>lb)[0][[0,-1]]
    zmin, zmax = np.where(z>lb)[0][[0,-1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def _load(x):
    '''
    Load the data from the path to the memory (not VRAM)
    '''
    img_t1, img_t2, basename = x
    preprocessed_path = op.join(data_dir, "preprocessed_h5")
    if os.path.exists(op.join(preprocessed_path, basename + '.h5')):
        return
    try:
        subject = tio.Subject(
            image_t1 = tio.ScalarImage(img_t1),
            image_t2 = tio.ScalarImage(img_t2),
        )
        transform_1 = tio.Compose([
            tio.transforms.HistogramStandardization(landmarks_dict),
            tio.transforms.RescaleIntensity((0., 1.)),
            tio.transforms.ToCanonical(),
            tio.transforms.Resample(spacing),
            tio.transforms.RandomBlur((1.5,1.5,1.5,1.5,2,2)),
            tio.transforms.Resample((1.,1.,1.)),
        ])
        transform_1_gt = tio.Compose([
            tio.transforms.HistogramStandardization(landmarks_dict),
            tio.transforms.RescaleIntensity((0., 1.)),
            tio.transforms.ToCanonical(),
            tio.transforms.Resample((1.,1.,1.)),
        ])
        subject_gt = transform_1_gt(subject)
        subject = transform_1(subject)
        shape = subject_gt.image_t2.data.numpy()[0].shape
        lb = np.percentile(subject.image_t2.data.numpy(),1)
        bbox = get_bbox(subject.image_t2.data.numpy()[0],lb)
        transform_crop = tio.transforms.Crop((bbox[0], shape[0]-bbox[1], bbox[2], shape[1]-bbox[3], bbox[4], shape[2]-bbox[5]))
        subject = transform_crop(subject)
        subject_gt = transform_crop(subject_gt)
        edge_max = max(subject.image_t1.data.shape)
        padding = ((edge_max - subject.image_t1.data.shape[1]) // 2, 
                    (edge_max - subject.image_t1.data.shape[2]) // 2,
                        (edge_max - subject.image_t1.data.shape[3]) // 2)
        transform_2 = tio.Compose([
            tio.Pad(padding),
            tio.transforms.Resize((160,160,160)),
        ])
        subject = transform_2(subject)
        subject_gt = transform_2(subject_gt)
        if not op.exists(preprocessed_path):
            os.makedirs(preprocessed_path)
        with h5.File(op.join(preprocessed_path, basename + '.h5'), 'w') as f:
            f.create_dataset('image_t1', data=subject.image_t1.data[0])
            f.create_dataset('image_t2', data=subject.image_t2.data[0])
            f.create_dataset('gt_t1', data=subject_gt.image_t1.data[0])
            f.create_dataset('gt_t2', data=subject_gt.image_t2.data[0])
    except:
        return

def load(list_images_t1,list_images_t2,list_basenames):
    # Not recommended to use multiprocessing for unstable output
    # The torchio is already using multiprocessing
    thread_map(_load, zip(list_images_t1, list_images_t2,list_basenames), max_workers=1, total=num_samples)

load(list_images_t1,list_images_t2,list_basenames)