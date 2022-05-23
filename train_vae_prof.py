from datetime import date
import profile
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

from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from pytorch_memlab import MemReporter


import os
import time
import numpy as np



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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.log_to_file(os.path.join(
        output_dir, 'log_{}'.format(str(time.time()))))
    logger.info('Build data loader')
    n_splits = gpc.config.N_SPLITS if gpc.config.N_SPLITS is not None else 5
    if gpc.config.DATASET == 'dHCP':
        dataloaders, val_loader = get_synth_dhcp_dataloader(data_dir=gpc.config.DATA_DIR,
                                                            batch_size=1,
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
        # model = BUNet3D(in_channels=gpc.config.IN_CHANNELS,
        #                 out_channels=gpc.config.OUT_CHANNELS,
        #                 f_maps=gpc.config.F_MAPS,
        #                 layer_order='gczr',
        #                 num_groups=min(1, gpc.config.F_MAPS[0]//2),
        #                 is_segmentation=False,
        #                 latent_size=gpc.config.LATENT_SIZE,
        #                 alpha=gpc.config.ALPHA if gpc.config.ALPHA is not None else 0.00025)
        model = UNet3D(in_channels=gpc.config.IN_CHANNELS,
                        out_channels=gpc.config.OUT_CHANNELS,
                        f_maps=gpc.config.F_MAPS,
                        layer_order='gcr',
                        num_groups=min(1, gpc.config.F_MAPS[0]//2),
                        is_segmentation=False,
                        alpha=gpc.config.ALPHA if gpc.config.ALPHA is not None else 0.00025,
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
        # engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(
        #     model=model,
        #     optimizer=optim,
        #     lr_scheduler=lr_scheduler,
        #     criterion=criterion,
        #     train_dataloader=train_loader,
        #     test_dataloader=test_loader,
        #     verbose=True,)
        # logger.info("engine is built", ranks=[0])

        val_image,val_target = val_loader.dataset.__getitem__(0)
        if not os.path.exists(os.path.join(output_dir, '{}'.format(i))):
            os.makedirs(os.path.join(output_dir, '{}'.format(i)))
        np.save(os.path.join(output_dir,'{}'.format(i), 'val_image.npy'), val_image)
        np.save(os.path.join(output_dir,'{}'.format(i), 'val_target.npy'), val_target)

        # trainer = Trainer(engine=engine, logger=logger)
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

        for img,gt in tqdm(train_loader):
            img=img.cuda()
            gt=gt.cuda()
            model=model.cuda()
            optim.zero_grad()
            with profile(activities=[ProfilerActivity.CUDA],
                    profile_memory=True, record_shapes=True) as prof:
                output = model(img)
            loss = criterion(output, gt)
            loss.backward()
            optim.step()
            lr_scheduler.step()
            reporter = MemReporter()
            reporter.report()
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            break


if __name__ == '__main__':
    train()
