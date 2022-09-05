import torch
import sys
sys.path.append("/media/hdd/viscent/SR-UNet")
from models.unet3d.model import BUNet3D,UNet3D

import os 
import numpy as np
import SimpleITK as sitk
import ants
import matplotlib.pyplot as plt
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchviz import make_dot
from colossalai.utils import load_checkpoint
from colossalai.initialize import launch,initialize
import colossalai
from colossalai.trainer import Trainer, hooks
import h5py as h5
from dataloaders import get_synth_dhcp_dataloader, get_synth_hcp_dataloader, get_synth_brats_dataloader
import torchio as tio
from tqdm.notebook import tqdm

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space.
def display_image(image_z, image):
    img = image[:, :, image_z]
    plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis("off")
    plt.show()

def display_multiplanar(image, x=1, y=1, z=1):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(image[x,:,:], cmap=plt.cm.Greys_r)
    ax1.axis("off")
    ax2.imshow(image[:,y,:], cmap=plt.cm.Greys_r)
    ax2.axis("off")
    ax3.imshow(image[:,:,z], cmap=plt.cm.Greys_r)
    ax3.axis("off")
    plt.show()
    return fig

def display_multiplanar_center(image):
    return display_multiplanar(image, x=image.shape[0]//2, y=image.shape[1]//2, z=image.shape[2]//2)

def display_images(image_z, image_0, image_1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image_0[:, :, image_z], cmap=plt.cm.Greys_r)
    ax1.axis("off")
    ax2.imshow(image_1[:, :, image_z], cmap=plt.cm.Greys_r)
    ax2.axis("off")
    

def load_model(model_path:str):
    model = torch.load(model_path)
    model.train()
    return model

def get_dataloader(dataset='dhcp', num_samples=None, modality='t1'):
    if dataset == 'dhcp':
        data_dir = '/media/hdd/dhcp/dhcp_lores/'
        loaders, val_loader = get_synth_dhcp_dataloader(data_dir=data_dir,
                                                                batch_size=1,
                                                                num_samples=num_samples,
                                                                input_modalities=[modality],
                                                                output_modalities=[modality],
                                                                output_dir=data_dir,
                                                                n_splits=5,
                                                                augmentation=False,
                                                                down_factor=5,)
        train_loader, test_loader = loaders[0]
        return train_loader, test_loader, val_loader
    elif dataset == 'hcp':
        data_dir = '/media/hdd/HCP_1200'
        loaders, val_loader = get_synth_hcp_dataloader(data_dir=data_dir,
                                                                batch_size=1,
                                                                num_samples=num_samples,
                                                                input_modalities=[modality],
                                                                output_modalities=[modality],
                                                                output_dir=data_dir,
                                                                n_splits=5,
                                                                augmentation=False,
                                                                down_factor=5,)
        train_loader, test_loader = loaders[0]
        return train_loader, test_loader, val_loader
    elif dataset == 'brats':
        data_dir = '/media/hdd/BraTS2020/'
        loaders, val_loader = get_synth_brats_dataloader(data_dir=data_dir,
                                                                batch_size=1,
                                                                num_samples=num_samples,
                                                                input_modalities=[modality],
                                                                output_modalities=[modality],
                                                                output_dir=data_dir,
                                                                n_splits=5,
                                                                augmentation=False,
                                                                down_factor=5,)
        train_loader, test_loader = loaders[0]
        return train_loader, test_loader, val_loader

def load_image(mode :str = 'h5', 
               paths = ['/home/viscent/hdd/dhcp/dhcp_lores/preprocessed_h5/sub-CC00582XX14.h5'],
               dataset = 'dhcp',
               data_dir = '/home/viscent/hdd/dhcp/dhcp_lores/',
               modality = 't1',):
    if mode == 'h5':
        with h5.File(paths[0],'r') as f:
            image = f['image_{modality}'.format(modality=modality)][...].astype(np.float32)
            target = f['gt_{modality}'.format(modality=modality)][...].astype(np.float32)
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
                                                                input_modalities=[modality],
                                                                output_modalities=[modality],
                                                                output_dir=data_dir,
                                                                n_splits=5,
                                                                augmentation=False,
                                                                down_factor=5,)
        elif dataset == 'hcp':
            dataloaders, val_loader = get_synth_hcp_dataloader(data_dir=data_dir,
                                                                batch_size=1,
                                                                num_samples=50,
                                                                input_modalities=[modality],
                                                                output_modalities=[modality],
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

def get_metrics(output_tensor,target_tensor, model, mu_q = None, logvar_q = None):
    im = target_tensor
    im_hat = output_tensor
    mse = torch.nn.MSELoss()(im, im_hat)
    mse.backward()
    print('mse:', mse.cpu().detach().numpy())
    if hasattr(model, 'enc_mu'):
        # mu_p, logvar_p = model.enc_mu,model.enc_logvar
        # # kl = (0.5 * ((torch.ones_like(logvar_p)-torch.ones_like(logvar_p)) + (mu_p-mu_q)**2 / torch.ones_like(logvar_p).exp() - 1 + (torch.ones_like(logvar_p)).exp() / (torch.ones_like(logvar_p)).exp() )).sum()
        # # kl = (0.5 * ((logvar_q-logvar_p) + (mu_p-mu_q)**2 / logvar_q.exp() - 1 + logvar_p.exp() / logvar_q.exp())).mean()
        # kl = 0.5 * (logvar_p.exp() + mu_p**2 - 1 - logvar_p).sum()
        # # kl = 0.5 * ((logvar_q-logvar_p) - 3 + (mu_p - mu_q) / logvar_q.exp() * (mu_p-mu_q) + torch.trace)
        # # kl = torch.sum(kl)

        ###### kl divergence with covariance ######
        # mu_p = model.enc_mu.flatten()
        # logvar_p = model.enc_logvar.flatten()

        # cov_p = torch.exp(logvar_p)
        # cov_p[cov_p.isnan()] = cov_p.mean()
        # cov_p[cov_p == 0] = cov_p.mean()
        # cov_p = cov_p.diag()
        # cov_q = torch.exp(logvar_q)
        # cov_q[cov_q.isnan()] = cov_q.mean()
        # cov_q[cov_q == 0] = cov_q.mean()
        # cov_q = cov_q.diag()
        # k = mu_p.shape[0]

        # tmp = ((mu_p-mu_q)**2 / torch.exp(logvar_q))
        # tmp[tmp.isnan()] = tmp.median()
        # tmp[tmp.isinf()] = tmp.median()
        # tmp = tmp.sum()
        # kl = 0.5 *( logvar_q.sum() - logvar_p.sum() - k + tmp + (torch.exp(logvar_p-logvar_q).sum()))
        kl = torch.sum(0.5 * (torch.exp(logvar) + torch.pow(mu,2) - 1 - logvar))
        ###########################################


        FE_simple = mse + 0.00025 * kl
        print('kl:', kl.cpu().detach().numpy())
        print('Free energy:', FE_simple.cpu().detach().numpy())

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

import matplotlib
def rgb_white2alpha(rgb, ensure_increasing=False, ensure_linear=False, lsq_linear=False):
    """
    Convert a set of RGB colors to RGBA with maximum transparency.

    The transparency is maximised for each color individually, assuming
    that the background is white.

    Parameters
    ----------
    rgb : array_like shaped (N, 3)
        Original colors.
    ensure_increasing : bool, default=False
        Ensure that alpha values increase monotonically.
    ensure_linear : bool, default=False
        Ensure alpha values increase linear from initial to final value.
    lsq_linear : bool, default=False
        Use least-squares linear fit for alpha.

    Returns
    -------
    rgba : numpy.ndarray shaped (N, 4)
        Colors with maximum possible transparency, assuming a white
        background.
    """
    # The most transparent alpha we can use is given by the min of RGB
    # Convert it from saturation to opacity
    alpha = 1. - np.min(rgb, axis=1)
    if lsq_linear:
        # Make a least squares fit for alpha
        indices = np.arange(len(alpha))
        A = np.stack([indices, np.ones_like(indices)], axis=-1)
        m, c = np.linalg.lstsq(A, alpha, rcond=None)[0]
        # Use our least squares fit to generate a linear alpha
        alpha = c + m * indices
        alpha = np.clip(alpha, 0, 1)
    elif ensure_linear:
        # Use a linearly increasing/decreasing alpha from start to finish
        alpha = np.linspace(alpha[0], alpha[-1], rgb.shape[0])
    elif ensure_increasing:
        # Let's also ensure the alpha value is monotonically increasing
        a_max = alpha[0]
        for i, a in enumerate(alpha):
            alpha[i] = a_max = np.maximum(a, a_max)
    alpha = np.expand_dims(alpha, -1)
    # Rescale colors to discount the white that will show through from transparency
    rgb = (rgb + alpha - 1)
    rgb = np.divide(rgb, alpha, out=np.zeros_like(rgb), where=(alpha > 0))
    rgb = np.clip(rgb, 0, 1)
    # Concatenate our alpha channel
    rgba = np.concatenate((rgb, alpha), axis=1)
    return rgba
def cmap_white2alpha(name, ensure_increasing=False, ensure_linear=False, lsq_linear=False, register=True):

    """
    Add as much transparency as possible to a colormap, assuming white background.

    Parameters
    ----------
    name : str
        Name of builtin (or registered) colormap.
    ensure_increasing : bool, default=False
        Ensure that alpha values are strictly increasing.
    ensure_linear : bool, default=False
        Ensure alpha values increase linear from initial to final value.
    lsq_linear : bool, default=False
        Use least-squares linear fit for alpha.
    register : bool, default=True
        Whether to register the new colormap.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap with alpha set as low as possible.
    """
    # Fetch the cmap callable
    cmap = plt.get_cmap(name)
    # Get the colors out from the colormap LUT
    rgb = cmap(np.arange(cmap.N))[:, :3]  # N-by-3
    # Convert white to alpha
    rgba = rgb_white2alpha(
        rgb,
        ensure_increasing=ensure_increasing,
        ensure_linear=ensure_linear,
        lsq_linear=lsq_linear,
    )
    # Create a new Colormap object
    new_name = name + "_white2alpha"
    cmap_alpha = matplotlib.colors.ListedColormap(rgba, name=new_name)
    if register:
        matplotlib.cm.register_cmap(name=new_name, cmap=cmap_alpha)
    return cmap_alpha

def eval(image_tensor, target_tensor, model,output_name):
    image_tensor = image_tensor.cuda()
    target_tensor = target_tensor.cuda()
    output_tensor = model(image_tensor)
    model.VAE_loss(output_tensor, target_tensor)
    kl = model.kl
    mse = model.mse
    print('KL Divergence:', kl)
    print('MSE:', mse)
    image_array = image_tensor.cpu().detach().squeeze(0).squeeze(0).numpy()
    output_array = output_tensor.cpu().detach().squeeze(0).squeeze(0).numpy()
    target_array = target_tensor.cpu().detach().squeeze(0).squeeze(0).numpy()
    image_fig = display_multiplanar_center(image_array)
    output_fig = display_multiplanar_center(output_array)
    target_fig = display_multiplanar_center(target_array)
    image_fig.savefig(output_name+'_image.png')
    output_fig.savefig(output_name+'_output.png')
    target_fig.savefig(output_name+'_target.png')
    ## make dir if not exist
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    sitk.WriteImage(sitk.GetImageFromArray(image_array), os.path.join(output_name, 'image.nii'))
    sitk.WriteImage(sitk.GetImageFromArray(output_array), os.path.join(output_name, 'output.nii'))
    sitk.WriteImage(sitk.GetImageFromArray(target_array), os.path.join(output_name, 'target.nii'))

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
    
def kl_forward(self, x):
    with torch.no_grad():
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        # VAE part
        # x = x.view(x.size(0),-1,self.latent_size)
        encoders_features = encoders_features[1:]
        x = torch.transpose(x, 1, 4)

        mu = self.mu(x)
        logvar = self.logvar(x)
        kl = torch.sum(0.5 * (torch.exp(logvar) + torch.pow(mu,2) - 1 - logvar))
        return kl

def kl_forward_prior(self, x, mu_q, logvar_q):
    with torch.no_grad():
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        # VAE part
        # x = x.view(x.size(0),-1,self.latent_size)
        encoders_features = encoders_features[1:]
        x = torch.transpose(x, 1, 4)

        mu = self.mu(x).flatten()
        logvar = self.logvar(x).flatten()
        mu_p = mu
        logvar_p = logvar

        # cov_p = torch.exp(logvar_p)
        # cov_p[cov_p.isnan()] = cov_p.mean()
        # cov_p[cov_p == 0] = cov_p.mean()
        # cov_p = cov_p.diag()
        # cov_q = torch.exp(logvar_q)
        # cov_q[cov_q.isnan()] = cov_q.mean()
        # cov_q[cov_q == 0] = cov_q.mean()
        # cov_q = cov_q.diag()
        k = mu_p.shape[0]

        # tmp = ((mu_p-mu_q)**2 / torch.exp(logvar_q))
        # tmp[tmp.isnan()] = tmp.median()
        # tmp[tmp.isinf()] = tmp.median()
        # tmp = tmp.sum()
        logvar_q = logvar_q.double()
        logvar_p = logvar_p.double()
        mu_q = mu_q.double()
        mu_p = mu_p.double()
        tmp = ((mu_p-mu_q)**2 / torch.exp(logvar_q))
        tmp = tmp.sum()
        kl = 0.5 *( logvar_q.sum() - logvar_p.sum() - k + tmp + (torch.exp(logvar_p-logvar_q).sum()))
        # kl = 0.5 *( logvar_q.sum() - logvar_p.sum() - k + tmp + (torch.exp(logvar_p-logvar_q).sum()))
        # tmp = (logvar_q - logvar_p + torch.exp(2*(logvar_p - logvar_q))/2 - 0.5)
        # tmp[tmp.isnan()] = tmp.median()
        # tmp[tmp.isinf()] = tmp.median()
        # tmp = torch.sigmoid(tmp)
        # tmp = (tmp-tmp.mean())/tmp.std()
        # kl = tmp.mean()
        # kl = torch.nn.functional.softmax(kl, dim=0)
        # print(kl)
        return kl

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# Gaussian blur kernel
def get_gaussian_kernel(device="cpu"):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(device)
    return gaussian_k

def pyramid_down(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)        
    # channel-wise conv(important)
    multiband = [F.conv2d(image[:, i:i + 1,:,:], gaussian_k, padding=2, stride=2) for i in range(3)]
    down_image = torch.cat(multiband, dim=1)
    return down_image

def pyramid_up(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)
    upsample = F.interpolate(image, scale_factor=2)
    multiband = [F.conv2d(upsample[:, i:i + 1,:,:], gaussian_k, padding=2) for i in range(3)]
    up_image = torch.cat(multiband, dim=1)
    return up_image

def gaussian_pyramid(original, n_pyramids, device="cpu"):
    x = original
    # pyramid down
    pyramids = [original]
    for i in range(n_pyramids):
        x = pyramid_down(x, device=device)
        pyramids.append(x)
    return pyramids

def laplacian_pyramid(original, n_pyramids, device="cpu"):
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids, device=device)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - pyramid_up(pyramids[i + 1], device=device)
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])        
    return laplacian

def minibatch_laplacian_pyramid(image, n_pyramids, batch_size, device="cpu"):
    n = image.size(0) // batch_size + np.sign(image.size(0) % batch_size)
    pyramids = []
    for i in range(n):
        x = image[i * batch_size:(i + 1) * batch_size]
        p = laplacian_pyramid(x.to(device), n_pyramids, device=device)
        p = [x.cpu() for x in p]
        pyramids.append(p)
    del x
    result = []
    for i in range(n_pyramids + 1):
        x = []
        for j in range(n):
            x.append(pyramids[j][i])
        result.append(torch.cat(x, dim=0))
    return result

def extract_patches(pyramid_layer, slice_indices,
                    slice_size=7, unfold_batch_size=128, device="cpu"):
    assert pyramid_layer.ndim == 4
    n = pyramid_layer.size(0) // unfold_batch_size + np.sign(pyramid_layer.size(0) % unfold_batch_size)
    # random slice 7x7
    p_slice = []
    for i in range(n):
        # [unfold_batch_size, ch, n_slices, slice_size, slice_size]
        ind_start = i * unfold_batch_size
        ind_end = min((i + 1) * unfold_batch_size, pyramid_layer.size(0))
        x = pyramid_layer[ind_start:ind_end].unfold(
                2, slice_size, 1).unfold(3, slice_size, 1).reshape(
                ind_end - ind_start, pyramid_layer.size(1), -1, slice_size, slice_size)
        # [unfold_batch_size, ch, n_descriptors, slice_size, slice_size]
        x = x[:,:, slice_indices,:,:]
        # [unfold_batch_size, n_descriptors, ch, slice_size, slice_size]
        p_slice.append(x.permute([0, 2, 1, 3, 4]))
    # sliced tensor per layer [batch, n_descriptors, ch, slice_size, slice_size]
    x = torch.cat(p_slice, dim=0)
    # normalize along ch
    std, mean = torch.std_mean(x, dim=(0, 1, 3, 4), keepdim=True)
    x = (x - mean) / (std + 1e-8)
    # reshape to 2rank
    x = x.reshape(-1, 3 * slice_size * slice_size)
    return x
        
def swd(image1, image2, 
        n_pyramids=None, slice_size=7, n_descriptors=128,
        n_repeat_projection=128, proj_per_repeat=4, device="cpu", return_by_resolution=False,
        pyramid_batchsize=128):
    # n_repeat_projectton * proj_per_repeat = 512
    # Please change these values according to memory usage.
    # original = n_repeat_projection=4, proj_per_repeat=128    
    assert image1.size() == image2.size()
    assert image1.ndim == 4 and image2.ndim == 4

    if n_pyramids is None:
        n_pyramids = int(np.rint(np.log2(image1.size(2) // 16)))
    with torch.no_grad():
        # minibatch laplacian pyramid for cuda memory reasons
        pyramid1 = minibatch_laplacian_pyramid(image1, n_pyramids, pyramid_batchsize, device=device)
        pyramid2 = minibatch_laplacian_pyramid(image2, n_pyramids, pyramid_batchsize, device=device)
        result = []

        for i_pyramid in range(n_pyramids + 1):
            # indices
            n = (pyramid1[i_pyramid].size(2) - 6) * (pyramid1[i_pyramid].size(3) - 6)
            indices = torch.randperm(n)[:n_descriptors]

            # extract patches on CPU
            # patch : 2rank (n_image*n_descriptors, slice_size**2*ch)
            p1 = extract_patches(pyramid1[i_pyramid], indices,
                            slice_size=slice_size, device="cpu")
            p2 = extract_patches(pyramid2[i_pyramid], indices,
                            slice_size=slice_size, device="cpu")

            p1, p2 = p1.to(device), p2.to(device)

            distances = []
            for j in range(n_repeat_projection):
                # random
                rand = torch.randn(p1.size(1), proj_per_repeat).to(device)  # (slice_size**2*ch)
                rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
                # projection
                proj1 = torch.matmul(p1, rand)
                proj2 = torch.matmul(p2, rand)
                proj1, _ = torch.sort(proj1, dim=0)
                proj2, _ = torch.sort(proj2, dim=0)
                d = torch.abs(proj1 - proj2)
                distances.append(torch.mean(d))

            # swd
            result.append(torch.mean(torch.stack(distances)))
        
        # average over resolution
        result = torch.stack(result) * 1e3
        if return_by_resolution:
            return result.cpu()
        else:
            return torch.mean(result).cpu()

def eval_hyperfine(image_path: str, model, mu, logvar):
    image = sitk.ReadImage(image_path)
    image_tensor = torch.from_numpy(sitk.GetArrayFromImage(image))
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).float().cuda()
    with torch.no_grad():
        output_tensor = model(image_tensor)
    fig_in = display_multiplanar_center(image_tensor[0,0].cpu().numpy())
    fig_out = display_multiplanar_center(output_tensor[0,0].cpu().numpy())
    kl = kl_forward_prior(model,output_tensor,mu,logvar).cpu().detach()
    return fig_in, fig_out, kl
def eval_tensor(image_tensor, model, mu, logvar):
    image_tensor = image_tensor.cuda()
    with torch.no_grad():
        output_tensor = model(image_tensor)
    display_multiplanar_center(image_tensor[0,0].cpu().numpy())
    display_multiplanar_center(output_tensor[0,0].cpu().numpy())
    kl = kl_forward_prior(model,output_tensor,mu,logvar).cpu().detach()
    print("KL: ",kl)