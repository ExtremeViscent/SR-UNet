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

class SynthdHCPDataset(Dataset):
    def __init__(self, data_dir, phase="train", num_samples=None, input_modalities=['t1'], output_modalities=['t1'], augmentation=False,down_factor=5):
        self.data_dir = data_dir
        self.phase = phase # train, val, test
        self.augmentation = augmentation 
        self.down_factor = down_factor # Previously for pre-processing, now deprecated
        self.input_dual_modal = True if len(input_modalities) == 2 else False
        self.output_dual_modal = True if len(output_modalities) == 2 else False
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        folder_name="preprocessed_h5"
        self.preprocessed_path = op.join(data_dir, folder_name)
        if op.exists(self.preprocessed_path):
            self.list_basenames = sorted(glob.glob(op.join(self.preprocessed_path, '*.h5')))
            self.list_basenames = [op.basename(x).split('.')[0] for x in self.list_basenames]
            self.list_images_t1 = self.list_basenames
            self.list_images_t2 = self.list_basenames
            self.list_labels = self.list_basenames
        else:
            self.list_labels = glob.glob(op.join(data_dir,'labels', '*.nii.gz'))
            self.list_labels.sort()
            self.list_images_t1 = [x.replace('_desc-drawem9_dseg.nii.gz', '_T1w_brain.nii.gz') for x in self.list_labels]
            self.list_images_t2 = [x.replace('_desc-drawem9_dseg.nii.gz', '_T2w_brain.nii.gz') for x in self.list_labels]
            self.list_images_t1 = [x.replace('_desc-drawem9_dseg_1mm.nii.gz', '_T1w_brain_1mm.nii.gz') for x in self.list_images_t1]
            self.list_images_t2 = [x.replace('_desc-drawem9_dseg_1mm.nii.gz', '_T2w_brain_1mm.nii.gz') for x in self.list_images_t2]
            self.list_images_t1 = [x.replace(op.join(data_dir, 'labels'), op.join(data_dir, 'images_t1')) for x in self.list_images_t1]
            self.list_images_t2 = [x.replace(op.join(data_dir, 'labels'), op.join(data_dir, 'images_t2')) for x in self.list_images_t2]
            self.list_basenames = [op.basename(x).split('_')[0] for x in self.list_images_t1]



        assert (num_samples is None) or num_samples >= 10, "num_samples must be >= 10"

        if num_samples is None:
            self.num_samples = len(self.list_labels)
        else:
            self.num_samples = num_samples
        if not self.list_labels:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        if phase == "train":
            self.list_images_t1 = self.list_images_t1[:int(0.8 * self.num_samples)]
            self.list_images_t2 = self.list_images_t2[:int(0.8 * self.num_samples)]
            self.list_labels = self.list_labels[:int(0.8 * self.num_samples)]
            self.list_basenames = self.list_basenames[:int(0.8 * self.num_samples)]
            self.num_samples = int(0.8 * self.num_samples)
        elif phase == "val":
            self.list_images_t1 = self.list_images_t1[int(0.8 * self.num_samples):self.num_samples]
            self.list_images_t2 = self.list_images_t2[int(0.8 * self.num_samples):self.num_samples]
            self.list_labels = self.list_labels[int(0.8 * self.num_samples):self.num_samples]
            self.list_basenames = self.list_basenames[int(0.8 * self.num_samples):self.num_samples]
            self.num_samples = int(0.2 * self.num_samples)
        logging.info(f'Creating dataset with {self.num_samples} examples')
        logging.info(f'length of list_images_t1: {len(self.list_images_t1)}')

        # TODO - delete this##################

        spacing = [1.0,1.0,1.0]
        spacing = np.array(spacing)
        spacing *= down_factor
        target_shape = (217, 290, 290)
        factor = spacing[2] / spacing[0]
        aniso_transform = tio.RandomAnisotropy(axes=2, downsampling=factor)
        resize_transform = tio.Resize(target_shape=target_shape)
        resample_transform = tio.Resample(target=spacing)
        self.transform  = tio.Compose([resample_transform,resize_transform])
        ####################################

        self.transform_spatial_1 = K.RandomAffine3D((15., 20., 20.),translate=(0.1,0.1,0.2), p=0.4,keepdim=True)
        self.transform_intensity_1 = K.RandomMotionBlur3D(5, 5., 0.5, p=0.4,keepdim=True)
        self.transform_intensity_2 = lambda x: x + torch.randn_like(x)*np.percentile(x, 99)*0.03 if np.random.rand() < 0.4 else x

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
            # Parameters generated from here:
            image = self.transform_spatial_1(image)
            # image = self.transform_spatial_2(image)
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
        img_t1, img_t2, gt, basename = x
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
                # if self.input_dual_modal:
                #     image = np.stack(f['image_t1'][:], f['image_t2'][:])
                # else:
                #     image = np.expand_dims(f['image_{}'.format(self.input_modalities[0])][:], axis=0)
                # if self.output_dual_modal:
                #     gt = np.stack(f['gt_t1'][:], f['gt_t2'][:])
                # else:
                #     gt = np.expand_dims(f['gt_{}'.format(self.output_modalities[0])][:], axis=0)
                image = image.astype(np.float32)
                gt = gt.astype(np.float32)
            return image, gt

        #######TODO - delete this##################
        image_t1 = sitk.ReadImage(img_t1)
        image_t2 = sitk.ReadImage(img_t2)
        gt_t1 = image_t1
        gt_t2 = image_t2
        if self.augmentation:
            image_t1 = self.transform(image_t1)
            image_t2 = self.transform(image_t2)
            # gt_t1 = self.transform_gt(gt_t1)
            # gt_t2 = self.transform_gt(gt_t2)
        image_t1 = sitk.GetArrayFromImage(image_t1)
        image_t2 = sitk.GetArrayFromImage(image_t2)
        gt_t1 = sitk.GetArrayFromImage(gt_t1)
        if self.input_dual_modal:
            image = np.stack([image_t1, image_t2], axis=0)
        elif self.input_modalities[0] == 't1':
            image = np.expand_dims(image_t1, axis=0)
        elif self.input_modalities[0] == 't2':
            image = np.expand_dims(image_t2, axis=0)
        if self.output_dual_modal:
            gt = np.stack([gt_t1, gt_t2], axis=0)
        elif self.output_modalities[0] == 't1':
            gt = np.expand_dims(gt_t1, axis=0)
        elif self.output_modalities[0] == 't2':
            gt = np.expand_dims(gt_t2, axis=0)
        image = image.astype(np.float32)
        gt = gt.astype(np.float32)
        with h5.File(op.join(self.preprocessed_path, basename + '.h5'), 'w') as f:
            f.create_dataset('image', data=image)
            f.create_dataset('gt', data=gt)
        return image, gt
        ###########################################

    def load(self):
        # Load images into memory
        ret = thread_map(self._load, zip(self.list_images_t1, self.list_images_t2, self.list_labels,self.list_basenames), max_workers=1, total=self.num_samples)
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
# TODO - delete this
class SynthdHCPDatasetTIO(tio.SubjectsDataset):
    def __init__(self, data_dir, phase="train", num_samples=None, input_modalities=['t1'], output_modalities=['t1'], augmentation=False,down_factor=5):
        self.data_dir = data_dir
        self.phase = phase
        self.augmentation = augmentation
        self.down_factor = down_factor
        self.input_dual_modal = True if len(input_modalities) == 2 else False
        self.output_dual_modal = True if len(output_modalities) == 2 else False
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        folder_name="preprocessed"
        self.preprocessed_path = op.join(data_dir, folder_name)
        assert op.exists(self.preprocessed_path), "Path {} does not exist".format(self.preprocessed_path)
        self.list_basenames = sorted(glob.glob(op.join(self.preprocessed_path, '*_t1.nii.gz')))
        self.list_basenames = [op.basename(x).split('_')[0] for x in self.list_basenames]
        assert (num_samples is None) or num_samples >= 10, "num_samples must be >= 10"

        if num_samples is None:
            self.num_samples = len(self.list_basenames)
        else:
            self.num_samples = num_samples
        if not self.list_basenames:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        if phase == "train":
            self.list_basenames = self.list_basenames[:int(0.8 * self.num_samples)]
            self.num_samples = int(0.8 * self.num_samples)
        elif phase == "val":
            self.list_basenames = self.list_basenames[int(0.8 * self.num_samples):self.num_samples]
            self.augmentation = False
            self.num_samples = int(0.2 * self.num_samples)
        logging.info(f'Creating dataset with {self.num_samples} examples')
        logging.info(f'length of list_images_t1: {len(self.list_basenames)}')

        spacing = [1.0,1.0,1.0]
        spacing = np.array(spacing)
        spacing *= 8
        target_shape = (108, 145, 145)
        factor = spacing[2] / spacing[0]
        aniso_transform = tio.RandomAnisotropy(axes=2, downsampling=10)
        resize_transform = tio.Resize(target_shape=target_shape)
        resample_transform = tio.Resample(target=spacing)
        blur_transform = tio.RandomBlur(6)
        augmentation_transform = []
        if self.augmentation:
            augmentation_transform.append(tio.RandomFlip(axes=(0,1)))
            augmentation_transform.append(tio.RandomAffine())
            augmentation_transform.append(tio.RandomElasticDeformation())
            augmentation_transform.append(tio.RandomMotion())
            augmentation_transform.append(tio.RandomBiasField())
            augmentation_transform.append(tio.RandomSwap())
            # augmentation_transform.append(resample_transform)
            # augmentation_transform.append(resize_transform)
            # augmentation_transform.append(blur_transform)
            augmentation_transform = tio.Compose(augmentation_transform)
            self.transform.keep = {'t1': 'gt_t1', 't2': 'gt_t2'}
        else:
            self.augmentation = None
            
        self.transform  = augmentation_transform
        self.subjects=[]


        self.load()
        if self.augmentation:
            super().__init__(self.subjects, transform=self.transform,load_getitem=True)
        else:
            super().__init__(self.subjects,load_getitem=True)




    def __getitem__(self, idx):
        # with h5.File(self.preprocessed_path + f"/{self.list_basenames[idx]}.h5", 'r') as f:
        #     image = f['image'][:]
        #     gt = f['gt'][:]
        # print('getting item'+str(idx))
        subject = super().__getitem__(idx)

        images = []
        gts = []
        for modality in self.input_modalities:
            images.append(subject[modality].data.numpy()[0])
        for modality in self.output_modalities:
            gts.append(subject['gt_'+modality].data.numpy()[0])
        image = np.array(images).astype(np.float32)
        gt = np.array(gts).astype(np.float32)
        # print('item'+str(idx)+' done')
        return image, gt




    def _load(self,basename):

        img_t1_pth = op.join(self.data_dir, 'preprocessed',basename+'_t1.nii.gz')
        img_t2_pth = op.join(self.data_dir, 'preprocessed',basename+'_t2.nii.gz')
        gt_t1_pth = op.join(self.data_dir, 'preprocessed',basename+'_gt_t1.nii.gz')
        gt_t2_pth = op.join(self.data_dir, 'preprocessed',basename+'_gt_t2.nii.gz')
        img_t1 = tio.ScalarImage(img_t1_pth)
        img_t2 = tio.ScalarImage(img_t2_pth)
        gt_t1 = tio.ScalarImage(gt_t1_pth)
        gt_t2 = tio.ScalarImage(gt_t2_pth)
        subject = tio.Subject(
            t1 = img_t1,
            t2 = img_t2,
            gt_t1 = gt_t1,
            gt_t2 = gt_t2,
            id = basename,
        )
        return subject

        

    def load(self):
        ret = thread_map(self._load,self.list_basenames, max_workers=32, total=self.num_samples)
        self.subjects = ret