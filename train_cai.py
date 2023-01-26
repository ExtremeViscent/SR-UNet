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
from colossalai.utils.checkpointing import save_checkpoint
from torch.optim.lr_scheduler import CosineAnnealingLR


from dataloaders import get_synth_dhcp_dataloader, get_synth_hcp_dataloader, get_synth_brats_dataloader
from models.unet3d.model import BUNet3D, UNet3D
import importlib
import SimpleITK as sitk
from PIL import Image
from geomloss import SamplesLoss

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

class BetaScheduler():
    def __init__(self, model, min=0,max=0.0001, cycle_len=1000):
        self.model = model
        self.min = min
        self.max = max
        self.current_step = 0
        self.cycle_len = cycle_len
    def get_beta(self):
        return self.model.alpha
    def step(self):
        self.model.alpha = self.min + (self.max - self.min) * (1 - np.cos(self.current_step / self.cycle_len * np.pi)) / 2
        self.current_step += 1 

class TrainLoggerHook(colossalai.trainer.hooks.BaseHook):
    def __init__(self, priority: int, TBLogger):
        super().__init__(priority)
        self.TBLogger = TBLogger
    def after_train_iter(self, trainer, output, label, loss):
        self.TBLogger(phase='train', step=trainer.cur_step,loss=loss)

class EvalHook(colossalai.trainer.hooks.SaveCheckpointHook):
    def after_test_epoch(self, trainer):
        cur_epoch = trainer.cur_epoch
        output_dir = gpc.config.OUTPUT_DIR
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        pred_dir = os.path.join(output_dir, 'preds')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        ckpt_path = os.path.join(checkpoint_dir, '{}.pth'.format(cur_epoch))
        if cur_epoch % 10 == 0:
            save_checkpoint(op.join(checkpoint_dir,'{}.pth'.format(cur_epoch)), cur_epoch, self.model, trainer.engine.optimizer,self._lr_scheduler)
        image = np.load(os.path.join(output_dir, 'val_image.npy'))
        target = np.load(os.path.join(output_dir, 'val_target.npy'))
        image = torch.tensor(image).unsqueeze(0).cuda()
        target = torch.tensor(target).unsqueeze(0).cuda()
        pred = trainer.engine.model(image)
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
            im_target = target[0, 0, image.shape[2]//2, :, :]
            im_image = (im_image-np.min(im_image)) / \
                (np.max(im_image)-np.min(im_image))*255
            im_target = (im_target-np.min(im_target)) / \
                (np.max(im_target)-np.min(im_target))*255
            im_image = Image.fromarray(im_image).convert('RGB')
            im_target = Image.fromarray(im_target).convert('RGB')
            im_image.save(os.path.join(output_dir, 'image.png'))
            im_target.save(os.path.join(output_dir, 'target.png'))
        # sitk.WriteImage(sitk.GetImageFromArray(pred[0, 0, :, :, :]),
        #                 os.path.join(pred_dir, '{}.nii.gz'.format(cur_epoch)))
        if not os.path.exists(os.path.join(pred_dir, '2d')):
            os.makedirs(os.path.join(pred_dir, '2d'))
        if cur_epoch % 10 == 0:
            im_pred.save(os.path.join(pred_dir, "2d",
                        '{}.png'.format(cur_epoch)))

def train():
    # Debug: find anormaly 
    torch.autograd.set_detect_anomaly(True)
    # Initialize Colossal-AI context
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    # Use default config and port
    if args.config is None:
        args.config = './config.py'
    port = 11451
    success = False
    # Find a free port
    while not success:
        try:
            colossalai.launch(args.config,0,1,'localhost',port)
            success = True
        except:
            port+=1
    # Initialize the logger
    logger = get_dist_logger()
    # Create paths
    output_dir = gpc.config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info('Build data loader')
    # For KFold cross validation, not needed for now
    n_splits = gpc.config.N_SPLITS if gpc.config.N_SPLITS is not None else 5
    # Initialize data loaders
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
    elif gpc.config.DATASET == 'BraTS':
        dataloaders, val_loader = get_synth_brats_dataloader(data_dir=gpc.config.DATA_DIR,
                                                            batch_size=gpc.config.BATCH_SIZE,
                                                            num_samples=gpc.config.NUM_SAMPLES,
                                                            input_modalities=gpc.config.INPUT_MODALITIES,
                                                            output_modalities=gpc.config.OUTPUT_MODALITIES,
                                                            output_dir=output_dir,
                                                            n_splits=n_splits,
                                                            augmentation=gpc.config.AUGMENTATION,
                                                            down_factor=gpc.config.DOWN_FACTOR,)
    
    train_loader, test_loader = dataloaders[0]
    # Define model, optimizer and loss
    model = UNet3D(in_channels=gpc.config.IN_CHANNELS,
                    out_channels=gpc.config.OUT_CHANNELS,
                    f_maps=gpc.config.F_MAPS,
                    layer_order='gcr',
                    num_groups=min(1, gpc.config.F_MAPS[0]//2),
                    is_segmentation=False,
                    )
    criterion = nn.MSELoss()
    logger.info('Using MSE loss')
    optim = torch.optim.Adam(
        model.parameters(),
        lr=gpc.config.LR,
        betas=(0.9, 0.999)
    )
    lr_scheduler = CosineAnnealingLR(optim, 1000)
    # Initialize the trainer
    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(
        model,
        optim, 
        criterion, 
        train_loader, 
        test_loader, 
        lr_scheduler
    )
    trainer = Trainer(engine, logger=logger)
    # Initialize the TB Logger
    TBLogger = TensorBoardLogger(log_dir = op.join(output_dir, 'tb_logs'))
    TBHook = TrainLoggerHook(100, TBLogger)
    # Saving validation images
    val_image,val_target = val_loader.dataset.__getitem__(0)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, 'val_image.npy'), val_image)
    np.save(os.path.join(output_dir, 'val_target.npy'), val_target)

    
    trainer.fit(train_dataloader, 
    gpc.config.NUM_EPOCHS, 
    None, 
    test_dataloader,
    1,
    [TBHook, 
    EvalHook(101, op.join(output_dir,'checkpoints','best_model.pth'))],
    True)

            

if __name__ == '__main__':
    train()
