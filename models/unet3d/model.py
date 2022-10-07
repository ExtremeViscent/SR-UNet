from copy import deepcopy
from functools import reduce
import torch.nn as nn
import torch
import torch.optim as optim
from torchmetrics import StructuralSimilarityIndexMeasure
from piqa import SSIM
from torchmetrics.functional import structural_similarity_index_measure
from geomloss import SamplesLoss

from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger

from .buildingblocks import DoubleConv, ExtResNetBlock, create_encoders, \
    create_decoders
from .utils import number_of_features_per_level, get_class

import kornia.augmentation as K

class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(Abstract3DUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(
                f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class Abstract3DBUNet(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, latent_size=32, alpha=0.8, augmentation=True, recon_loss_func='mse', div_loss_func='kl', **kwargs):
        super(Abstract3DBUNet, self).__init__(in_channels, out_channels, final_sigmoid, basic_module, f_maps,
                                              layer_order, num_groups, num_levels, is_segmentation, conv_kernel_size,
                                              pool_kernel_size, conv_padding, **kwargs)
        self.alpha = alpha
        self.logger = get_dist_logger()
        self.enc_mu = None
        self.enc_logvar = None
        self.warming_up = True
        self.div_loss = None
        self.recon_loss = None
        self.recon_loss_func = recon_loss_func
        self.div_loss_func = div_loss_func
        self.latent_size = latent_size
        self.augmentation = augmentation
        self.mu = nn.Sequential(
            nn.Linear(f_maps[-1], latent_size),
            nn.LayerNorm(latent_size)
        )
        self.logvar = nn.Sequential(
            nn.Linear(f_maps[-1], latent_size),
            nn.LayerNorm(latent_size)
        )
        self.latent_to_decode = nn.Sequential(
            nn.Linear(latent_size, f_maps[-1]),
            nn.LayerNorm(f_maps[-1]),
        )
        # Junk for now
        self.transform = nn.Sequential(
            K.RandomRotation3D((15., 20., 20.), p=0.5,keepdim=True),
            K.RandomMotionBlur3D(3, 35., 0.5, p=0.4,keepdim=True),
            K.RandomAffine3D((15., 20., 20.), p=0.4,keepdim=True),
        )
        self.init_weights()

    def forward(self, x):
        # encoder part
        # Line below is for debugging, junk for now
        if self.augmentation:
            x = self.transform(x)
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

        
        # self.kl = None
        # self.mse = None

        self.enc_mu = nn.Parameter(mu)
        self.enc_logvar = nn.Parameter(logvar)
        sample = self.sample_from_mu_var(mu, logvar)
        self.latent = nn.Parameter(sample)
        # sample = MultivariateNormal(mu, torch.exp(logvar))
        x = self.latent_to_decode(sample)
        x = torch.transpose(x, 1, 4)
        # encoders_features.insert(2, x)
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        # TODO: delete this
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def enable_fe_loss(self):
        self.warming_up = False

    def sample_from_mu_var(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mu)
        return sample

    def init_weights(self):
        nn.init.eye_(self.mu[0].weight.data)
        nn.init.zeros_(self.mu[0].bias.data)
        nn.init.normal_(self.logvar[0].weight.data, 0, 0.1)
        nn.init.zeros_(self.logvar[0].bias.data)

    def patch_mse(self, im,im_hat, kernel_size=8, stride=4):
        se = (im - im_hat)**2
        se = se.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
        max_ = se.max(-1).values.max(-1).values.max(-1).values
        min_ = se.min(-1).values.min(-1).values.min(-1).values
        mse = se.mean(-1).mean(-1).mean(-1)
        mse = (mse - min_)/(max_ - min_)
        mse = torch.mean(mse)
        return mse


    def VAE_loss(self, im, im_hat):
        recon_loss = 0.
        div_loss = 0.

        if self.warming_up:
            recon_loss = torch.nn.MSELoss()(im, im_hat)
            self.recon_loss = nn.Parameter(recon_loss, requires_grad=False)
            return recon_loss
        mu, logvar = self.enc_mu, self.enc_logvar
        if self.recon_loss_func == 'mse':
            recon_loss = torch.nn.MSELoss()(im, im_hat)
        elif self.recon_loss_func == 'ssim':
            recon_loss = 1 - structural_similarity_index_measure(im, im_hat)
        elif self.recon_loss_func == 'patch_mse':
            recon_loss = self.patch_mse(im, im_hat)
        self.recon_loss = nn.Parameter(recon_loss,requires_grad=False)

        if self.div_loss_func == 'kl':
            div_loss = torch.sum(0.5 * (torch.exp(logvar) + torch.pow(mu,2) - 1 - logvar))
            self.div_loss = nn.Parameter(div_loss,requires_grad=False)
        elif self.div_loss_func == 'sinkhorn':
            div_loss = SamplesLoss("sinkhorn")(self.latent.flatten(1), torch.zeros_like(self.latent.flatten(1)))
            self.div_loss = nn.Parameter(div_loss,requires_grad=False)
        return recon_loss + self.alpha * div_loss


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, alpha=0.00025, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)
        self.alpha = alpha
        self.kl = None
        self.mse = None
        self.warmup = True

    def enable_VAE(self):
        for encoder in self.encoders:
            vae_block = encoder.basic_module.SingleConv1.VAE
            vae_block.enable()
            vae_block = encoder.basic_module.SingleConv2.VAE
            vae_block.enable()      
        for decoder in self.decoders:
            vae_block = decoder.basic_module.SingleConv1.VAE
            vae_block.enable()
            vae_block = decoder.basic_module.SingleConv2.VAE
            vae_block.enable()
        self.warmup = False

    # def forward(self, x):
    #     if self.is_VAE:
    #         return self.VAE_forward(x)
    #     else:
    #         return super(UNet3D, self).forward(x)

    def VAE_loss(self, im, im_hat):
        kl = 0
        mse = torch.nn.MSELoss()(im, im_hat)
        if self.warmup:
            self.mse = mse
            return mse
        for encoder in self.encoders:
            _kl = encoder.get_metrics()
            kl+=_kl
        kl = kl/len(self.encoders)
        for decoder in self.decoders:
            _kl = decoder.get_metrics()
            kl+=_kl
        kl = kl/len(self.decoders)
        self.kl = kl
        self.mse = mse
        return mse + self.alpha * kl


class BUNet3D(Abstract3DBUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(BUNet3D, self).__init__(in_channels=in_channels,
                                      out_channels=out_channels,
                                      final_sigmoid=final_sigmoid,
                                      basic_module=DoubleConv,
                                      f_maps=f_maps,
                                      layer_order=layer_order,
                                      num_groups=num_groups,
                                      num_levels=num_levels,
                                      is_segmentation=is_segmentation,
                                      conv_padding=conv_padding,
                                      **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             **kwargs)


class UNet2D(Abstract3DUNet):
    """
    Just a standard 2D Unet. Arises naturally by specifying conv_kernel_size=(1, 3, 3), pool_kernel_size=(1, 2, 2).
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        if conv_padding == 1:
            conv_padding = (0, 1, 1)
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=(1, 3, 3),
                                     pool_kernel_size=(1, 2, 2),
                                     conv_padding=conv_padding,
                                     **kwargs)


def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
                            'pytorch3dunet.unet3d.model'])
    return model_class(**model_config)
