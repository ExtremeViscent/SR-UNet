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


import os
import time
import numpy as np

class VAESchedulerHook(colossalai.trainer.hooks.BaseHook):
    def __init__(self, warmup_epochs=0, priority=10, **kwargs):
        super().__init__(priority,**kwargs)
        self.warmup_epochs = warmup_epochs
        self.logger = get_dist_logger()
    def before_train_epoch(self, trainer):
        print(f"before_train_epoch {trainer.cur_epoch}")
        if trainer.cur_epoch == self.warmup_epochs:
            trainer.engine.model.model.enable_VAE()
            self.logger.info("Enabled VAE")


class AdvancedTBHook(hooks.TensorboardHook):
    def _log_by_iter(self, trainer, mode: str):
        super(AdvancedTBHook,self)._log_by_iter(trainer, mode)
        if self._is_valid_rank_to_log:
            model = trainer.engine.model.model
            for idx,encoder in enumerate(model.encoders):                    
                kl = encoder.get_metrics()
                if kl is not None:
                    self.writer.add_scalar(f'encoder{idx}/kl_loss', kl, trainer.cur_step)
                # self.writer.add_scalar(f'encoder{idx}/mse', mse, trainer.cur_step)
            for idx,decoder in enumerate(model.decoders):
                kl = decoder.get_metrics()
                if kl is not None:
                    self.writer.add_scalar(f'decoder{idx}/kl', kl, trainer.cur_step)
                # self.writer.add_scalar(f'decoder{idx}/mse', mse, trainer.cur_step)
            kl = model.kl
            mse = model.mse
            if kl is not None:
                self.writer.add_scalar(f'{mode}/kl', kl, trainer.cur_step)
            # self.writer.add_scalar(f'kl/{mode}', kl, trainer.cur_step)
            self.writer.add_scalar(f'mse/{mode}', mse, trainer.cur_step)

class SaveAndEvalByEpochHook(colossalai.trainer.hooks.BaseHook):
    def __init__(self, checkpoint_dir, output_dir,  fold, priority=10):
        super(SaveAndEvalByEpochHook, self).__init__(priority=priority)
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.fold = fold
        self.image_dir = os.path.join(self.output_dir, 'images')
        self.target_dir = os.path.join(self.output_dir, 'targets')
        self.pred_dir = os.path.join(self.output_dir, 'preds')
        if gpc.get_global_rank() == 0:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if not os.path.exists(self.pred_dir):
                os.makedirs(self.pred_dir)
            if not os.path.exists(os.path.join(self.pred_dir, '2d')):
                os.makedirs(os.path.join(self.pred_dir, '2d'))
        self.logger = get_dist_logger()

    # def after_test_iter(self, trainer, output, label, loss):
    #     model = trainer.engine.model
    #     kl, mse = model.get_metrics()
    #     self.logger.info('Epoch: {} MSE: {} KL: {}'.format(trainer.cur_epoch, mse, kl))


    def after_train_epoch(self, trainer):
        model = trainer.engine.model
        if gpc.get_global_rank() == 0:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
        torch.save(dict(state_dict=model.state_dict()),
                   os.path.join(self.checkpoint_dir, '{}.pth'.format(trainer.cur_epoch)))
        image = np.load(os.path.join(self.output_dir, 'val_image.npy'))
        target = np.load(os.path.join(self.output_dir, 'val_target.npy'))
        image = torch.tensor(image).unsqueeze(0).cuda()
        target = torch.tensor(target).unsqueeze(0).cuda()
        model.eval()
        pred = trainer.engine(image)
        image = image.cpu().detach().numpy().astype(np.float32)
        target = target.cpu().detach().numpy().astype(np.float32)
        pred = pred.cpu().detach().numpy().astype(np.float32)
        im_pred = pred[0, 0, pred.shape[2]//2, :, :]
        im_pred = (im_pred-np.min(im_pred)) / \
            (np.max(im_pred)-np.min(im_pred))*255
        im_pred = Image.fromarray(im_pred).convert('RGB')
        if trainer.cur_epoch == 0:
            if gpc.config.IN_CHANNELS == 2:
                sitk.WriteImage(sitk.GetImageFromArray(image[0, 0, :, :, :]),
                                os.path.join(self.output_dir, 'image_t1.nii.gz'))
                sitk.WriteImage(sitk.GetImageFromArray(image[0, 1, :, :, :]),
                                os.path.join(self.output_dir, 'image_t2.nii.gz'))
            else:
                sitk.WriteImage(sitk.GetImageFromArray(image[0, 0, :, :, :]),
                                os.path.join(self.output_dir, 'image.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(target[0, :, :, :]),
                            os.path.join(self.output_dir, 'target.nii.gz'))
            im_image = image[0, 0, image.shape[2]//2, :, :]
            im_target = target[0,0, image.shape[2]//2, :, :]
            im_image = (im_image-np.min(im_image)) / \
                (np.max(im_image)-np.min(im_image))*255
            im_target = (im_target-np.min(im_target)) / \
                (np.max(im_target)-np.min(im_target))*255
            im_image = Image.fromarray(im_image).convert('RGB')
            im_target = Image.fromarray(im_target).convert('RGB')
            im_image.save(os.path.join(self.output_dir, 'image.png'))
            im_target.save(os.path.join(self.output_dir, 'target.png'))
        sitk.WriteImage(sitk.GetImageFromArray(pred[0, 0, :, :, :]),
                        os.path.join(self.pred_dir, '{}.nii.gz'.format(trainer.cur_epoch)))
        im_pred.save(os.path.join(self.pred_dir, "2d",
                     '{}.png'.format(trainer.cur_epoch)))


def train():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()


    if args.config is None:
        args.config = './config.py'
    

    if os.getenv("OMPI_COMM_WORLD_RANK") is not None:
        node_list = os.popen(node_cmd)
        host = node_list.read().split('\n')[0]
        node_list.close()
        print('host: {}'.format(host))
        colossalai.launch_from_openmpi(config=args.config,
                                       host=host,
                                       port=11451,
                                       seed=42)
    elif (os.getenv("LOCAL_RANK") is not None):
        colossalai.launch_from_torch(config=args.config)
    elif os.getenv('SLURM_JOB_NODELIST') is not None:
        node_cmd = 'scontrol show hostnames \"$SLURM_JOB_NODELIST\"'
        #execute command
        node_list = os.popen(node_cmd)
        host = node_list.read().split('\n')[0]
        node_list.close()
        print('host: {}'.format(host))
        colossalai.launch_from_slurm(
            config=args.config,
            host=host,
            port=11451,
            backend='nccl',
        )
    else:
        print("Launch mode not supported")
    logger = get_dist_logger()
    output_dir = gpc.config.OUTPUT_DIR
    if gpc.get_global_rank() == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    logger.log_to_file(os.path.join(
        output_dir, 'log_{}'.format(str(time.time()))))
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
    # model = UNet3D(in_channels=gpc.config.IN_CHANNELS,
    #                out_channels=gpc.config.OUT_CHANNELS,
    #                f_maps=gpc.config.F_MAPS,
    #                layer_order='gcr',
    #                num_groups=8,
    #                is_segmentation=False)

    # optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

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
        else:
            model = UNet3D(in_channels=gpc.config.IN_CHANNELS,
                            out_channels=gpc.config.OUT_CHANNELS,
                            f_maps=gpc.config.F_MAPS,
                            layer_order='gcr',
                            num_groups=min(1, gpc.config.F_MAPS[0]//2),
                            is_segmentation=False,
                            )
        criterion = model.VAE_loss
        # criterion = torch.nn.MSELoss
        logger.info('Initializing K-Fold', ranks=[0])
        optim = torch.optim.Adam(
            model.parameters(),
            lr=gpc.config.LR,
            betas=(0.9, 0.99)
        )
        lr_scheduler = CosineAnnealingLR(
            optim, gpc.config.NUM_EPOCHS*(train_loader.__len__()//gpc.config.BATCH_SIZE))
        engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(
            model=model,
            optimizer=optim,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            verbose=True,)
        logger.info("engine is built", ranks=[0])

        val_image,val_target = val_loader.dataset.__getitem__(0)
        if not os.path.exists(os.path.join(output_dir, '{}'.format(i))):
            os.makedirs(os.path.join(output_dir, '{}'.format(i)))
        np.save(os.path.join(output_dir,'{}'.format(i), 'val_image.npy'), val_image)
        np.save(os.path.join(output_dir,'{}'.format(i), 'val_target.npy'), val_target)

        hook_list = [
            hooks.LossHook(),
            hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
            hooks.LogMetricByEpochHook(logger),
            AdvancedTBHook(log_dir=os.path.join(output_dir,'logs','{}'.format(i))),
            hooks.SaveCheckpointHook(
                checkpoint_dir=os.path.join(output_dir, 'checkpoints', 'fold_{}.pt'.format(i)),
                model=model),
            hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
            SaveAndEvalByEpochHook(
                checkpoint_dir=os.path.join(output_dir, 'checkpoints', 'fold_{}'.format(i)),
                output_dir=os.path.join(output_dir,'{}'.format(i)),
                fold=i),
            # VAESchedulerHook(warmup_epochs=gpc.config.WARMUP_EPOCHS,),
        ]
        trainer = Trainer(engine=engine, logger=logger)
        # for epoch in range(gpc.config.NUM_EPOCHS):
        #     engine.train()
        #     for im,gt in train_loader:
        #         im=im.cuda()
        #         gt=gt.cuda()
        #         engine.zero_grad()
        #         output = engine(im)
        #         loss = criterion(output, gt)
        #         engine.backward(loss)
        #         engine.step()

        trainer.fit(
            train_dataloader=train_dataloader,
            hooks=hook_list,
            epochs=gpc.config.NUM_EPOCHS,
            test_dataloader=test_dataloader,
            test_interval=1,
            display_progress=True
        )


if __name__ == '__main__':
    train()
