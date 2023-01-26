
from datetime import date
import torch
import torch.nn as nn
import torch.nn.functional as F

import colossalai
import colossalai.utils as utils
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from torch.optim.lr_scheduler import CosineAnnealingLR


from dataloaders import get_synth_dhcp_dataloader, get_synth_hcp_dataloader, get_synth_brats_dataloader
from models.unet3d.model import BUNet3D, UNet3D
import SimpleITK as sitk
from PIL import Image
from geomloss import SamplesLoss

from torch.utils.tensorboard import SummaryWriter
import os
import os.path as op
import time
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt


# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

class TensorBoardLogger():
    def __init__(self, log_dir, **kwargs):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir, **kwargs)
        
    
    def __call__(self, phase, step, **kwargs):
        for key, value in kwargs.items():
            self.writer.add_scalar(f'{key}/{phase}', value, step)
        
    def __call__(self, metrics: dict, step):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

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

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose3d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose3d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d( ngf, nc, 4, 2, 1, bias=False),
            nn.BatchNorm3d(nc),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose3d( nc, nc, 36, 2, 1, bias=False),
            # state size. (nc) x 160 x 160
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv3d(nc, nc, 36, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 64 x 64
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv3d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def eval(model, nz, n_steps, output_dir):
    model.eval()
    noise = torch.randn(1, nz, 1, 1, 1).cuda()
    with torch.no_grad():
        fake = model(noise).detach().cpu()
        image = fake[0,0,80,:,:]
        image = Image.fromarray((image.numpy() * 255).astype(np.uint8), 'L')
        image.save(os.path.join(output_dir, f'fake_{n_steps}.png'))
        


def train():
    # Debug: find anormaly 
    torch.autograd.set_detect_anomaly(True)

    # Rewrite this part (TODO)
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()


    if args.config is None:
        args.config = './config.py'
    port = 11451
    success = False
    # Find a free port (TODO: delete this part)
    while not success:
        try:
            colossalai.launch(args.config,0,1,'localhost',port)
            success = True
        except:
            port+=1

    # TODO: Get rid of colossialai
    logger = get_dist_logger()

    output_dir = gpc.config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info('Build data loader')
    # For KFold cross validation, not needed for now
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
    for i in range(0, 5):
        logger.info('Training fold {}'.format(i), ranks=[0])
        train_loader, test_loader = dataloaders[i]
        
        ## Define models
        netG = Generator(ngpu).cuda()
        netG.apply(weights_init)
        netD = Discriminator(ngpu).cuda()
        netD.apply(weights_init)

        logger.info('Initializing K-Fold', ranks=[0])
        optimG = torch.optim.Adam(
            netG.parameters(),
            lr=gpc.config.LR,
            betas=(0.5, 0.999)
        )
        optimD = torch.optim.Adam(
            netD.parameters(),
            lr=gpc.config.LR,
            betas=(0.5, 0.999)
        )
        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(1, nz, 1, 1, 1).cuda()

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.
        lr_schedulerG = CosineAnnealingLR(optimG, 1000)
        lr_schedulerD = CosineAnnealingLR(optimD, 1000)
        TBLogger = TensorBoardLogger(log_dir = op.join(output_dir,'{}'.format(i), 'tb_logs'),comment='fold_{}'.format(i))

        n_step = 0
        n_step_test = 0


        for epoch in range(gpc.config.NUM_EPOCHS):
            netG.cuda()
            netD.cuda()
            for _,gt in tqdm(train_loader):
                netG.train()
                netD.train()
                gt=gt.cuda()
                netD.zero_grad()
                label = torch.full((gt.size(0),), real_label).cuda()
                output = netD(gt).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(gt.size(0), nz, 1, 1, 1).cuda()
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimD.step()

                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimG.step()
                n_step += 1
                lr_schedulerG.step()
                lr_schedulerD.step()
                TBLogger({'loss_D':errD.item(),'loss_G':errG.item()},step=n_step)
                eval(netG, nz, n_step, output_dir)
            logger.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, gpc.config.NUM_EPOCHS, i, len(train_loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), ranks=[0])
        

            

if __name__ == '__main__':
    print(torch.cuda.is_available())
    train()
