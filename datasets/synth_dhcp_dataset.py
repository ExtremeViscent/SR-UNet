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

class SynthdHCPDataset(Dataset):
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

        spacing = [1.0,1.0,1.0]
        spacing = np.array(spacing)
        spacing *= down_factor
        target_shape = (217, 290, 290)
        factor = spacing[2] / spacing[0]
        aniso_transform = tio.RandomAnisotropy(axes=2, downsampling=factor)
        resize_transform = tio.Resize(target_shape=target_shape)
        resample_transform = tio.Resample(target=spacing)
        self.transform  = tio.Compose([resample_transform,resize_transform])
        # self.transform_gt = resize_transform

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





    def _load(self,x):
        img_t1, img_t2, gt, basename = x
        if op.exists(op.join(self.preprocessed_path, basename + '.h5')):
            with h5.File(op.join(self.preprocessed_path, basename + '.h5'), 'r') as f:
                if self.input_dual_modal:
                    image = np.stack(f['image_t1'][:], f['image_t2'][:])
                else:
                    image = np.expand_dims(f['image_{}'.format(self.input_modalities[0])][:], axis=0)
                if self.output_dual_modal:
                    gt = np.stack(f['gt_t1'][:], f['gt_t2'][:])
                else:
                    gt = np.expand_dims(f['gt_{}'.format(self.output_modalities[0])][:], axis=0)
            return image, gt
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
        

    def load(self):
        # image_save_path = op.join(self.data_dir, 'images_{}_{}.npy'.format(self.input_modalities, self.output_modalities))

        # pool = mp.Pool(mp.cpu_count()-16)
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
        # np.save(image_save_path, images)
        # np.save(gt_save_path, gts)



        # for idx in trange(0, self.max_id) if phase == "train" else trange(self.max_id-self.num_samples, self.max_id):
        #     if os.path.exists(op.join(data_dir, str(f"{idx:04d}") + '_image_t1.nii.gz')):
        #         image = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_image_t1.nii.gz'))
        #         gt = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_target.nii.gz'))
        #         # label = sitk.ReadImage(op.join(data_dir, str(idx) + '_label.nii.gz'))
        #         images.append(np.expand_dims(sitk.GetArrayFromImage(image), axis=0))
        #         gts.append(sitk.GetArrayFromImage(gt))
        #         num_samples+=1
        # self.num_samples = num_samples
        # images = np.array(images)
        # gts = np.array(gts)
        # self.images = images
        # self.gts = gts

    # def load_aug(self, data_dir, phase="train"):
    #     images = []
    #     gts = []
    #     num_samples=0
    #     spacing = (1.5, 1.5, 5.0)
    #     factor = spacing[2] / spacing[0]
    #     transform = tio.RandomAnisotropy(axes=2, downsampling=factor)
    #     for idx in trange(0, self.max_id) if phase == "train" else trange(self.max_id-self.num_samples, self.max_id):
    #         if os.path.exists(op.join(data_dir, str(f"{idx:04d}") + '_image_t1.nii.gz')):
    #             image = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_image_t1.nii.gz'))
    #             images.append(np.expand_dims(sitk.GetArrayFromImage(image), axis=0))
    #             gts.append(np.expand_dims(sitk.GetArrayFromImage(image), axis=0))
    #             num_samples+=1
    #     self.num_samples = num_samples
    #     images = np.array(images)
    #     gts = np.array(gts)
    #     self.images = images
    #     self.gts = gts


    # def load_dual_modal(self, data_dir, phase="train"):
    #     images = []
    #     gts = []
    #     num_samples=0
    #     for idx in trange(0, self.max_id) if phase == "train" else trange(self.max_id-self.num_samples, self.max_id):
    #         if os.path.exists(op.join(data_dir, str(f"{idx:04d}") + '_image_t1.nii.gz')):
    #             image_t1 = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_image_t1.nii.gz'))
    #             image_t2 = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_image_t2.nii.gz'))
    #             image = np.stack([sitk.GetArrayFromImage(image_t1), sitk.GetArrayFromImage(image_t2)], axis=0)
    #             gt = sitk.ReadImage(op.join(data_dir, str(f"{idx:04d}") + '_target.nii.gz'))
    #             # label = sitk.ReadImage(op.join(data_dir, str(idx) + '_label.nii.gz'))
    #             images.append(image)
    #             gts.append(sitk.GetArrayFromImage(gt))
    #             num_samples+=1
    #     self.num_samples = num_samples
    #     images = np.array(images)
    #     gts = np.array(gts)
    #     self.images = images
    #     self.gts = gts