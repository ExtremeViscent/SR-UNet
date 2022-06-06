from cmath import phase
from datetime import date
from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F

import colossalai
import colossalai.utils as utils
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.schedule import (InterleavedPipelineSchedule,
                                        PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.nn.lr_scheduler import CosineAnnealingLR


from dataloaders import get_synth_dhcp_dataloader, get_synth_hcp_dataloader
from models.unet3d.model import BUNet3D, UNet3D
import importlib
import SimpleITK as sitk
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import os
import os.path as op
import time
from tqdm import tqdm
import numpy as np

class TensorBoardLogger():
    def __init__(self, log_dir, **kwargs):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir, **kwargs)
        
    
    def __call__(self, phase, step, **kwargs):
        for key, value in kwargs.items():
            self.writer.add_scalar(f'{key}/{phase}', value, step)
        

def eval(model, cur_epoch,fold):
    output_dir = gpc.config.OUTPUT_DIR
    output_dir=os.path.join(output_dir,'{}'.format(fold))
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    pred_dir = os.path.join(output_dir, 'preds')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    ckpt_path = os.path.join(checkpoint_dir, '{}.pth'.format(cur_epoch))
    torch.save(dict(state_dict=model.state_dict()),ckpt_path)
    # model = UNet3D(in_channels=gpc.config.IN_CHANNELS,
    #                 out_channels=gpc.config.OUT_CHANNELS,
    #                 f_maps=gpc.config.F_MAPS,
    #                 layer_order='gcr',
    #                 num_groups=min(1, gpc.config.F_MAPS[0]//2),
    #                 is_segmentation=False,
    #                 )
    # ckpt = torch.load(ckpt_path)
    # model.load_state_dict(ckpt['state_dict'])
    image = np.load(os.path.join(output_dir, 'val_image.npy'))
    target = np.load(os.path.join(output_dir, 'val_target.npy'))
    image = torch.tensor(image).unsqueeze(0).cuda()
    target = torch.tensor(target).unsqueeze(0).cuda()
    model.eval()
    pred = model(image)
    image = image.cpu().detach().numpy().astype(np.float32)
    target = target.cpu().detach().numpy().astype(np.float32)
    pred = pred.cpu().detach().numpy().astype(np.float32)
    im_pred = pred[0, 0, pred.shape[2]//2, :, :]
    im_pred = (im_pred-np.min(im_pred)) / \
        (np.max(im_pred)-np.min(im_pred))*255
    im_pred = Image.fromarray(im_pred).convert('RGB')
    if cur_epoch == 0:
        if gpc.config.IN_CHANNELS == 2:
            sitk.WriteImage(sitk.GetImageFromArray(image[0, 0, :, :, :]),
                            os.path.join(output_dir, 'image_t1.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(image[0, 1, :, :, :]),
                            os.path.join(output_dir, 'image_t2.nii.gz'))
        else:
            sitk.WriteImage(sitk.GetImageFromArray(image[0, 0, :, :, :]),
                            os.path.join(output_dir, 'image.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(target[0, :, :, :]),
                        os.path.join(output_dir, 'target.nii.gz'))
        im_image = image[0, 0, image.shape[2]//2, :, :]
        im_target = target[0,0, image.shape[2]//2, :, :]
        im_image = (im_image-np.min(im_image)) / \
            (np.max(im_image)-np.min(im_image))*255
        im_target = (im_target-np.min(im_target)) / \
            (np.max(im_target)-np.min(im_target))*255
        im_image = Image.fromarray(im_image).convert('RGB')
        im_target = Image.fromarray(im_target).convert('RGB')
        im_image.save(os.path.join(output_dir, 'image.png'))
        im_target.save(os.path.join(output_dir, 'target.png'))
    sitk.WriteImage(sitk.GetImageFromArray(pred[0, 0, :, :, :]),
                    os.path.join(pred_dir, '{}.nii.gz'.format(cur_epoch)))
    if not os.path.exists(os.path.join(pred_dir, '2d')):
        os.makedirs(os.path.join(pred_dir, '2d'))
    im_pred.save(os.path.join(pred_dir, "2d",
                    '{}.png'.format(cur_epoch)))

def train():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()


    if args.config is None:
        args.config = './config.py'
    colossalai.launch(args.config,0,1,'localhost',11451)
    logger = get_dist_logger()
    output_dir = gpc.config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info('Build data loader')
    n_splits = gpc.config.N_SPLITS if gpc.config.N_SPLITS is not None else 5
    if gpc.config.DATASET == 'dHCP':
        dataloaders, val_loader = get_synth_dhcp_dataloader(data_dir=gpc.config.DATA_DIR,
                                                            batch_size=gpc.config.BATCH_SIZE,
                                                            num_samples=gpc.config.NUM_SAMPLES,
                                                            input_modalities=gpc.config.INPUT_MODALITIES,
                                                            output_modalities=gpc.config.OUTPUT_MODALITIES,
                                                            output_dir=output_dir,
                                                            n_splits=n_splits,
                                                            augmentation=gpc.config.AUGMENTATION,
                                                            down_factor=gpc.config.DOWN_FACTOR,)
    elif gpc.config.DATASET == 'HCP':
        dataloaders, val_loader = get_synth_hcp_dataloader(data_dir=gpc.config.DATA_DIR,
                                                            batch_size=gpc.config.BATCH_SIZE,
                                                            num_samples=gpc.config.NUM_SAMPLES,
                                                            input_modalities=gpc.config.INPUT_MODALITIES,
                                                            output_modalities=gpc.config.OUTPUT_MODALITIES,
                                                            output_dir=output_dir,
                                                            n_splits=n_splits,
                                                            augmentation=gpc.config.AUGMENTATION,
                                                            down_factor=gpc.config.DOWN_FACTOR,)
    for i in range(0, 5):
        logger.info('Training fold {}'.format(i), ranks=[0])
        train_loader, test_loader = dataloaders[i]
        vae = getattr(gpc.config, 'VAE', True)
        if vae:
            model = BUNet3D(in_channels=gpc.config.IN_CHANNELS,
                            out_channels=gpc.config.OUT_CHANNELS,
                            f_maps=gpc.config.F_MAPS,
                            layer_order='gcr',
                            num_groups=min(1, gpc.config.F_MAPS[0]//2),
                            is_segmentation=False,
                            latent_size=gpc.config.LATENT_SIZE,
                            alpha=gpc.config.ALPHA if gpc.config.ALPHA is not None else 0.00025)
            criterion = model.VAE_loss
            logger.info('Using VAE loss')
        else:
            model = UNet3D(in_channels=gpc.config.IN_CHANNELS,
                            out_channels=gpc.config.OUT_CHANNELS,
                            f_maps=gpc.config.F_MAPS,
                            layer_order='gcr',
                            num_groups=min(1, gpc.config.F_MAPS[0]//2),
                            is_segmentation=False,
                            )
            criterion = torch.nn.MSELoss()
            logger.info('Using MSE loss')
        logger.info('Initializing K-Fold', ranks=[0])
        optim = torch.optim.Adam(
            model.parameters(),
            lr=gpc.config.LR,
            betas=(0.9, 0.99)
        )
        lr_scheduler = CosineAnnealingLR(
            optim, gpc.config.NUM_EPOCHS*(train_loader.__len__()//gpc.config.BATCH_SIZE))
        TBLogger = TensorBoardLogger(log_dir = op.join(output_dir,'{}'.format(i), 'tb_logs'),comment='fold_{}'.format(i))
        val_image,val_target = val_loader.dataset.__getitem__(0)
        if not os.path.exists(os.path.join(output_dir, '{}'.format(i))):
            os.makedirs(os.path.join(output_dir, '{}'.format(i)))
        np.save(os.path.join(output_dir,'{}'.format(i), 'val_image.npy'), val_image)
        np.save(os.path.join(output_dir,'{}'.format(i), 'val_target.npy'), val_target)

        model.cuda()
        n_step = 0
        for epoch in range(gpc.config.NUM_EPOCHS):
            model.train()
            for im,gt in tqdm(train_loader):
                im=im.cuda()
                gt=gt.cuda()
                optim.zero_grad()
                output = model(im)
                loss = criterion(output, gt)
                TBLogger(phase='train', step=epoch,loss=loss)
                loss.backward()
                optim.step()
            logger.info('Epoch {}/{}'.format(epoch, gpc.config.NUM_EPOCHS), ranks=[0])
            logger.info('Train Loss: {:.4f}'.format(loss.item()), ranks=[0])
            model.eval()
            with torch.no_grad():
                for im,gt in tqdm(test_loader):
                    im=im.cuda()
                    gt=gt.cuda()
                    output = model(im)  
                    loss = criterion(output, gt)
                    TBLogger(phase='test', step=epoch,loss=loss)
            logger.info('epoch:{}/{}'.format(epoch,gpc.config.NUM_EPOCHS), ranks=[0])
            logger.info('Test loss:{}'.format(loss), ranks=[0])
            eval(model,epoch,i)

            

if __name__ == '__main__':
    train()
