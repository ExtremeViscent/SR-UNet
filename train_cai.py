from datetime import date
from sched import scheduler
from turtle import forward
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


from dataloaders import get_synth_dhcp_dataloader
from models.unet3d.model import BUNet3D, UNet3D
import importlib
import SimpleITK as sitk
from PIL import Image


import os
import time
import numpy as np
import hiddenlayer as hl
from torchviz import make_dot


class custom_MSE(torch.nn.MSELoss):
    def forward(self, input, target):
        input = input.squeeze(1)
        return super(custom_MSE, self).forward(input, target)

class SaveAndEvalByEpochHook(colossalai.trainer.hooks.BaseHook):
    def __init__(self, checkpoint_dir, output_dir, dataloader, fold, priority=10):
        super(SaveAndEvalByEpochHook, self).__init__(priority=priority)
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.dataloader = dataloader
        self.fold = fold
        self.image_dir = os.path.join(self.output_dir, 'images')
        self.target_dir = os.path.join(self.output_dir, 'targets')
        self.pred_dir = os.path.join(self.output_dir, 'preds')
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
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(dict(state_dict=model.state_dict()),
                   os.path.join(self.checkpoint_dir, '{}.pth'.format(trainer.cur_epoch)))
        image, target = self.dataloader.dataset.__getitem__(0)
        image = torch.tensor(image).unsqueeze(0).cuda()
        target = torch.tensor(target).unsqueeze(0).cuda()
        _image = image
        model.eval()
        pred = trainer.engine(image)
        _pred = pred
        image = image.cpu().detach().numpy().astype(np.float32)
        target = target.cpu().detach().numpy().astype(np.float32)
        pred = pred.cpu().detach().numpy().astype(np.float32)
        im_pred = pred[0, 0, 48, :, :]
        im_pred = (im_pred-np.min(im_pred)) / \
            (np.max(im_pred)-np.min(im_pred))*255
        im_pred = Image.fromarray(im_pred).convert('RGB')
        if trainer.cur_epoch == 0:
            # graph = hl.build_graph(model, _image)
            # graph.theme = hl.graph.THEMES['blue'].copy()
            # graph.save(os.path.join(self.output_dir, 'graph.png'),format='png')
            make_dot(_pred,params=dict(model.named_parameters())).render(os.path.join(self.output_dir, 'graph_1.png'))
            if gpc.config.IN_CHANNELS == 2:
                sitk.WriteImage(sitk.GetImageFromArray(image[0, 0, :, :, :]),
                                os.path.join(self.output_dir, 'image_t1.nii.gz'))
                sitk.WriteImage(sitk.GetImageFromArray(image[0, 1, :, :, :]),
                                os.path.join(self.output_dir, 'image_t2.nii.gz'))
            else:
                sitk.WriteImage(sitk.GetImageFromArray(image[0, 0, :, :, :]),
                                os.path.join(self.output_dir, 'image.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(target[0, 0, :, :, :]),
                            os.path.join(self.output_dir, 'target.nii.gz'))
            im_image = image[0, 0, 48, :, :]
            im_target = target[0, 0, 48, :, :]
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
        model.train()

def get_dataloader(config):
    if config.DATASET == 'synth':
        return get_synth_dhcp_dataloader(config)
    else:
        raise NotImplementedError



def train():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    if args.config is None:
        args.config = './config.py'
    if os.getenv("OMPI_COMM_WORLD_RANK") is not None:
        colossalai.launch_from_openmpi(config=args.config,
                                       host=args.host,
                                       port=29500,
                                       seed=42)
    else:
        colossalai.launch_from_torch(config=args.config)
    logger = get_dist_logger()
    output_dir = gpc.config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.log_to_file(os.path.join(
        output_dir, 'log_{}'.format(str(time.time()))))
    logger.info('Build data loader')
    dataloaders, val_loader = get_synth_dhcp_dataloader(data_dir=gpc.config.DATA_DIR,
                                                        batch_size=gpc.config.BATCH_SIZE,
                                                        num_samples=50 if gpc.config.SMALL_DATA else None,
                                                        dual_modal=True if gpc.config.IN_CHANNELS == 2 else False,
                                                        output_dir=output_dir,)
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
        model = UNet3D(in_channels=gpc.config.IN_CHANNELS,
                        out_channels=gpc.config.OUT_CHANNELS,
                        f_maps=gpc.config.F_MAPS,
                        layer_order='gcr',
                        num_groups=8,
                        is_segmentation=False,
                        )

        # criterion = model.VAE_loss
        criterion = torch.nn.MSELoss()
        # criterion = custom_MSE()
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

        hook_list = [
            hooks.LossHook(),
            # hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
            hooks.LogMetricByEpochHook(logger),
            hooks.TensorboardHook(log_dir=os.path.join(output_dir,'logs','{}'.format(i))),
            hooks.SaveCheckpointHook(
                checkpoint_dir=os.path.join(output_dir, 'checkpoints', 'fold_{}.pt'.format(i)),
                model=model),
            hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
            SaveAndEvalByEpochHook(
                checkpoint_dir=os.path.join(output_dir, 'checkpoints', 'fold_{}'.format(i)),
                output_dir=os.path.join(output_dir,'{}'.format(i)),
                dataloader=val_loader,
                fold=i)
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
