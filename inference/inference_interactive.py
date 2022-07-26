import torch
import sys
sys.path.append("/scratch/users/k21113539/SR-UNet")
from models.unet3d.model import BUNet3D,UNet3D

import os 
import os.path as op
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import ants
from scipy import stats
from tqdm.notebook import tqdm,trange
import matplotlib.pyplot as plt
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchviz import make_dot
from colossalai.utils import load_checkpoint
from colossalai.initialize import launch,initialize
import colossalai
from colossalai.trainer import Trainer, hooks
import h5py as h5
from dataloaders import get_synth_dhcp_dataloader, get_synth_hcp_dataloader
import torchio as tio

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space.
def display_image(image_z, image):
    img = image[:, :, image_z]
    plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis("off")
    plt.show()

def load_model(model_path:str):
    model = torch.load(model_path)
    model.train()
    return model
def get_dataloader(dataset = 'dhcp', batch_size = 1, data_dir = '/scratch/prj/bayunet/dhcp_lores/'):
    if dataset == 'dhcp':
        dataloaders, val_loader = get_synth_dhcp_dataloader(data_dir=data_dir,
                                                            batch_size=batch_size,
                                                            num_samples=None,
                                                            input_modalities=["t1"],
                                                            output_modalities=["t1"],
                                                            output_dir=data_dir,
                                                            n_splits=5,
                                                            augmentation=False,
                                                            down_factor=5,)
    elif dataset == 'hcp':
        dataloaders, val_loader = get_synth_hcp_dataloader(data_dir=data_dir,
                                                            batch_size=batch_size,
                                                            num_samples=None,
                                                            input_modalities=["t1"],
                                                            output_modalities=["t1"],
                                                            output_dir=data_dir,
                                                            n_splits=5,
                                                            augmentation=False,
                                                            down_factor=5,)
    return val_loader
def load_image(mode :str = 'h5', 
               paths = ['/scratch/prj/bayunet/dhcp_lores/preprocessed_h5/sub-CC01104XX07.h5'],
               dataset = 'dhcp',
               data_dir = '/home/viscent/hdd/dhcp/dhcp_lores/',):
    if mode == 'h5':
        with h5.File(paths[0],'r') as f:
            image = f['image_t1'][...].astype(np.float32)
            target = f['gt_t1'][...].astype(np.float32)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).cuda()
        target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).cuda()
        return image_tensor,target_tensor
    elif mode == 'sitk':
        image = sitk.ReadImage(paths[0])
        target = sitk.ReadImage(paths[1])
        image = sitk.GetArrayFromImage(image)
        target = sitk.GetArrayFromImage(target)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).cuda()
        target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).cuda()
        return image_tensor,target_tensor
    elif mode == 'npy':
        image = np.load(paths[0])
        target = np.load(paths[1])
        image_tensor = torch.from_numpy(image).unsqueeze(0).cuda()
        target_tensor = torch.from_numpy(target).unsqueeze(0).cuda()
        return image_tensor,target_tensor
    elif mode == 'dataloader':
        if dataset == 'dhcp':
            dataloaders, val_loader = get_synth_dhcp_dataloader(data_dir=data_dir,
                                                                batch_size=1,
                                                                num_samples=50,
                                                                input_modalities=["t1"],
                                                                output_modalities=["t1"],
                                                                output_dir=data_dir,
                                                                n_splits=5,
                                                                augmentation=False,
                                                                down_factor=5,)
        elif dataset == 'hcp':
            dataloaders, val_loader = get_synth_hcp_dataloader(data_dir=data_dir,
                                                                batch_size=1,
                                                                num_samples=50,
                                                                input_modalities=["t1"],
                                                                output_modalities=["t1"],
                                                                output_dir=data_dir,
                                                                n_splits=5,
                                                                augmentation=False,
                                                                down_factor=5,)
        image_tensor, target_tensor = next(iter(val_loader))
        image_tensor = image_tensor.cuda()
        target_tensor = target_tensor.cuda()
        return image_tensor,target_tensor

def plot_latent(model):
    encoder_weights = next(model.encoders[-1].parameters())
    encoder_weights = encoder_weights.cpu().detach().numpy()
    encoder_weights= np.expand_dims(encoder_weights,axis=1)
    encoder_weights = np.repeat(encoder_weights, 128, axis=1)
    if hasattr(model, 'enc_mu'):
        fig,(ax1,ax2) = plt.subplots(1,2)
        im1 = ax1.imshow(encoder_weights)
        ax1.set_title("encoder weights")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, orientation="vertical")

        latent_weights = next(model.mu.parameters())
        latent_weights = latent_weights.cpu().detach().numpy()
        latent_weights = np.repeat(latent_weights, 128, axis=1)

        im2 = ax2.imshow(encoder_weights)
        ax2.set_title("latent weights")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax, orientation="vertical")
        fig.show()
    else:
        fig = plt.imshow(encoder_weights)
        plt.title("encoder weights")
        plt.colorbar()
        plt.show()
    # plt.savefig(OUTPUT_PREFIX+"_encoder_weights.png")

def infer(model,image_tensor):
    output_tensor = model(image_tensor)
    return output_tensor

def get_metrics(output_tensor,target_tensor, model, mu_q = None, logvar_q = None, verbose = True):
    im = target_tensor
    im_hat = output_tensor
    mse = torch.nn.MSELoss()(im, im_hat)
    mse.backward()
    if verbose:
        print('mse:', mse.cpu().detach().numpy())
    if hasattr(model, 'enc_mu'):
        mu_p, logvar_p = model.enc_mu,model.enc_logvar
        # kl = (0.5 * ((torch.ones_like(logvar_p)-torch.ones_like(logvar_p)) + (mu_p-mu_q)**2 / torch.ones_like(logvar_p).exp() - 1 + (torch.ones_like(logvar_p)).exp() / (torch.ones_like(logvar_p)).exp() )).sum()
        # kl = (0.5 * ((logvar_q-logvar_p) + (mu_p-mu_q)**2 / logvar_q.exp() - 1 + logvar_p.exp() / logvar_q.exp())).mean()
        kl = 0.5 * (logvar_p.exp() + mu_p**2 - 1 - logvar_p).sum()
        # kl = torch.mean(0.5 * (torch.exp(logvar_p) + torch.pow(mu_p,2) - 1 - logvar_p))
        # kl = 0.5 * ((logvar_q-logvar_p) - 3 + (mu_p - mu_q) / logvar_q.exp() * (mu_p-mu_q) + torch.trace)
        # kl = torch.sum(kl)
        FE_simple = mse + 0.00025 * kl
        if verbose:
            print('kl:', kl.cpu().detach().numpy())
            print('Free energy:', FE_simple.cpu().detach().numpy())
        return FE_simple, mse, kl
    else:
        return mse

def plot_output(image_tensor,output_tensor,target_tensor):
    image_tensor = image_tensor.cpu().squeeze().squeeze().detach().numpy().astype(np.float32)
    output_tensor = output_tensor.cpu().squeeze().squeeze().detach().numpy().astype(np.float32)
    target_tensor = target_tensor.cpu().squeeze().squeeze().detach().numpy().astype(np.float32)
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    im1 = ax1.imshow(image_tensor[image_tensor.shape[0]//2,:,:],cmap='gray')
    ax1.set_title("image")


    im2 = ax2.imshow(output_tensor[output_tensor.shape[0]//2,:,:],cmap='gray')
    ax2.set_title("output")
    
    
    im3 = ax3.imshow(target_tensor[target_tensor.shape[0]//2,:,:],cmap='gray')
    ax3.set_title("target")
    fig.show()
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# metric_values list.
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))
    
def registration_sitk(fixed_image, moving_image):
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    moving_resampled = sitk.Resample(
        moving_image,
        fixed_image,
        initial_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)


    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32)
    )
    moving_resampled = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )
    return moving_resampled

def resample(image):
    target_shape = (108, 145, 145)
    spacing = [1.0,1.0,1.0]
    spacing = np.array(spacing)
    resample_transform = tio.Resample(target=spacing)
    resize_transform = tio.Resize(target_shape=target_shape)
    transform  = tio.Compose([resample_transform,resize_transform])
    return transform(image)

def downSample(image):
    spacing = [1.0,1.0,1.0]
    spacing = np.array(spacing)
    spacing *= 5
    target_shape = (108, 145, 145)
    factor = spacing[2] / spacing[0]
    resize_transform = tio.Resize(target_shape=target_shape)
    resample_transform = tio.Resample(target=spacing)
    blur_transform = tio.RandomBlur(3)
    transform  = tio.Compose([resample_transform,resize_transform,blur_transform])
    return transform(image)


def registration_ants(fixed_image, moving_image):
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_array = sitk.GetArrayFromImage(moving_image)
    fixed_ants = ants.from_numpy(fixed_array)
    moving_ants = ants.from_numpy(moving_array)
    ret = ants.registration(fixed_ants, moving_ants,verbose=True)
    image = ret['warpedmovout'].numpy()
    image = sitk.GetImageFromArray(image)
    image.CopyInformation(fixed_image)
    return image

def main():
    model_path = input('Enter model path: ')

    model_path = Path(model_path)
    model = load_model(model_path)


    input_mode = input('Enter input mode (0: HDF5, 1: SITK, 2: npy, 3: dataloader): [0]') or '0'
    input_mode = int(input_mode)
    input_path = ''
    if input_mode == 0:
        input_path = input('Enter input path[default]: ')
        if not input_path:
            image_tensor,target_tensor = load_image()
        else:
            image_tensor,target_tensor = load_image(mode='h5',paths = [input_path])
    elif input_mode == 1:
        image_path = input('Enter input image path: ')
        gt_path = input('Enter input ground truth path: ')
        image_tensor,target_tensor = load_image(mode='sitk',paths = [image_path,gt_path])
    elif input_mode == 2:
        image_path = input('Enter input image path: ')
        gt_path = input('Enter input ground truth path: ')
        image_tensor,target_tensor = load_image(mode='npy',paths = [image_path,gt_path])
    elif input_mode == 3:
        input_dataset = input('Enter input dataset (0: dhcp, 1: hcp)[0]: ') or '0'
        input_dataset = 'dhcp' if input_dataset == '0' else 'hcp'
        input_path = input('Enter dataset directory: ')
        image_tensor,target_tensor = load_image(mode='dataloader', dataset = input_dataset, data_dir = input_path)
    else:
        print('Invalid path')
        return

    output_dir = input('Enter output directory: ')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    is_vae = False
    if hasattr(model, 'enc_mu'):
        mu_q = model.enc_mu
        logvar_q = model.enc_logvar
        is_vae = True

    # Inference

    output_tensor = model(image_tensor)
    image_array = image_tensor.cpu().squeeze().squeeze().detach().numpy().astype(np.float32)
    output_array = output_tensor.cpu().squeeze().squeeze().detach().numpy().astype(np.float32)
    target_array = target_tensor.cpu().squeeze().squeeze().detach().numpy().astype(np.float32)

    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_output = sitk.GetImageFromArray(output_array)
    sitk_target = sitk.GetImageFromArray(target_array)

    sitk.WriteImage(sitk_image, op.join(output_dir, 'image.nii.gz'))
    sitk.WriteImage(sitk_output, op.join(output_dir, 'output.nii.gz'))
    sitk.WriteImage(sitk_target, op.join(output_dir, 'target.nii.gz'))

    # Compute metrics

    summary = ''
    if is_vae:
        summary += 'VAE\n'
        FE_simple, mse, kl = get_metrics(output_tensor,target_tensor,model, mu_q= mu_q, logvar_q= logvar_q, verbose=False)
        summary += 'FE_simple: {:.4f}\n'.format(FE_simple)
        summary += 'mse: {:.4f}\n'.format(mse)
        summary += 'kl: {:.4f}\n'.format(kl)
    else:
        summary += 'U-Net\n'
        mse = get_metrics(output_tensor,target_tensor,model, verbose=False)
        summary += 'mse: {:.4f}\n'.format(mse)
    with open(op.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(summary)
    


if __name__ == '__main__':
    main()    