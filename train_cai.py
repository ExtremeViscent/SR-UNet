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


from dataloaders import get_synth_dhcp_dataloader
from models.unet3d.model import UNet3D
import importlib


import os 
import numpy as np

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
    logger.info('Build data loader')
    dataloaders = get_synth_dhcp_dataloader(data_dir = gpc.config.DATA_DIR,
                                            batch_size=gpc.config.BATCH_SIZE)
    # model = UNet3D(in_channels=gpc.config.IN_CHANNELS,
    #                out_channels=gpc.config.OUT_CHANNELS,
    #                f_maps=gpc.config.F_MAPS,
    #                layer_order='gcr',
    #                num_groups=8,
    #                is_segmentation=False)

    
    # optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    logger.info('Initializing K-Fold', ranks=[0])
    for i in range(0,5):
        logger.info('Training fold {}'.format(i), ranks=[0])
        train_loader, val_loader = dataloaders[i]
        model = UNet3D(in_channels=gpc.config.IN_CHANNELS,
                out_channels=gpc.config.OUT_CHANNELS,
                f_maps=gpc.config.F_MAPS,
                layer_order='gcr',
                num_groups=8,
                is_segmentation=False)
        criterion = torch.nn.MSELoss()
        optim = torch.optim.Adam(
            model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.99)
        )
        engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(
            model=model,
            optimizer=optim,
            criterion=criterion,
            train_dataloader=train_loader,
            test_dataloader=val_loader,
            verbose=True,)
        logger.info("engine is built", ranks=[0])


        hook_list = [
            hooks.LossHook(), 
            # hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True), 
            hooks.LogMetricByEpochHook(logger), 
            hooks.TensorboardHook(log_dir='./logs/{}'.format(i)), 
            hooks.SaveCheckpointHook(checkpoint_dir='./ckpt/{}'.format(i),model=model)]
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