from copy import deepcopy
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn import KLDivLoss
from  torch.distributions import MultivariateNormal

from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger

from .buildingblocks import DoubleConv, ExtResNetBlock, create_encoders, \
    create_decoders
from .utils import number_of_features_per_level, get_class
from torch.distributions import constraints


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
                 conv_padding=1, latent_size=32, alpha=0.8, **kwargs):
        super(Abstract3DBUNet, self).__init__(in_channels, out_channels, final_sigmoid, basic_module, f_maps,
                                              layer_order, num_groups, num_levels, is_segmentation, conv_kernel_size,
                                              pool_kernel_size, conv_padding, **kwargs)
        ## Input shape (B, 1, 108, 145, 145)
        ## BNL shape (B, 6, 9, 9, f_maps[-1])
        # self.mu = nn.Linear(f_maps[-1], latent_size)
        # self.logvar = nn.Linear(f_maps[-1], latent_size)
        self.bnl_size = 6 * 9 * 9 * f_maps[-1]
        self.mu = nn.Linear(self.bnl_size, latent_size)
        self.logvar = nn.Linear(self.bnl_size, latent_size ** 2)
        self.alpha = alpha
        self.logger = get_dist_logger()
        self.enc_mu = None
        self.enc_logvar = None
        self.warming_up = True
        self.kl = None
        self.mse = None
        self.latent_size = latent_size
        self.f_maps = f_maps
        self.latent_to_decode = nn.Linear(latent_size, self.bnl_size)
        self.init_weights()

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        # VAE part
        x = x.view(x.size(0),-1)
        encoders_features = encoders_features[1:]
        # x = torch.transpose(x, 1, 4)
        mu = self.mu(x)
        logvar = self.logvar(x)
        logvar = logvar.view(logvar.size(0), self.latent_size, self.latent_size)
        cov = torch.exp(logvar / 2)
        # Make it positive definite
        for b in range(cov.shape[0]):
            L,V = torch.linalg.eigh(cov[b])
            cov[b] = V.matmul(torch.diag(L)).matmul(V.inverse())
            L,V = torch.linalg.eigh(cov[b])
            print("original: ", cov[b])
            print("inverse transform:", V.matmul(torch.diag(L)).matmul(V.inverse()))
            s = 0
            p = 0
            for i in range(L.shape[0]):
                if L[i] < 0:
                    s += 1
                elif L[i] >0:
                    if p == 0:
                        p = L[i]
                    p = min(p, L[i])
            t = s**2 * 100 + 1
            for i in range(L.shape[0]):
                if L[i] < 0:
                    L[i] = p * (s-L[i]) ** 2 / t
            print("new eigenvalues:", L)
            cov[b] = V.matmul(torch.diag(L)).matmul(V.inverse())
            # cov[b] = V.matmul(torch.diag(L)).matmul(V.t())
            print(torch.linalg.eigh(cov[b]).eigenvalues)
            # cov[b] = torch.mm(cov[b], cov[b].t())
            # cov[b].add_(cov[b].clone().t()).div_(2)
            # print('covariance matrix: ', cov[b])
            print(constraints._PositiveDefinite().check(cov[b]))


        self.enc_mu = nn.Parameter(mu,requires_grad=False)
        self.enc_logvar = nn.Parameter(cov,requires_grad=False)
        # sample = self.sample_from_mu_var(mu, logvar)
        sample = MultivariateNormal(mu, cov).sample()
        x = self.latent_to_decode(sample)
        x = x.view(x.size(0), self.f_maps[-1], 9, 9, 6)
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
        nn.init.eye_(self.mu.weight.data)
        nn.init.zeros_(self.mu.bias.data)
        nn.init.normal_(self.logvar.weight.data, 0, 0.1)
        nn.init.zeros_(self.logvar.bias.data)


    def VAE_loss(self, im, im_hat):

        # im = im.squeeze()
        if self.warming_up:
            mse = torch.nn.MSELoss()(im, im_hat)
            self.mse = nn.Parameter(mse)
            return mse
        # mu, logvar = nn.functional.softmax(mu,dim=1), nn.functional.softmax(logvar,dim=1)
        # kl = torch.sum(0.5 * (torch.exp(logvar) + torch.pow(mu,2) - 1 - logvar))
        kl = torch.zeros(im.shape[0]).cuda()
        for i, (mu, cov) in enumerate(zip(self.enc_mu, self.enc_logvar)):
            kl_0 = mu.unsqueeze(0).matmul(mu.unsqueeze(0).t())
            kl_1 = torch.trace(cov).unsqueeze(0)
            kl_2 = mu.shape[0]
            kl_3 = torch.log(torch.det(cov)).unsqueeze(0)
            print(kl_0, kl_1, kl_2, kl_3)
            kl[i] = 0.5 * (kl_0 + kl_1 - kl_2 - kl_3)
        kl = torch.mean(kl)
        print('kl: ', kl)
        # kl = torch.mean(kl)
        # kl = KLDivLoss()(mu, logvar)
        # while kl * self.alpha > 1e+2:
        #     self.alpha *= 0.5
        # if self.kl is not None and kl/self.kl > 1000.:
        #         print('clipped kl {} and {}'.format(kl, self.kl))
        #         kl = self.kl
        # print("mu shape: ", mu.shape)
        # print("logvar shape: ", logvar.shape)
        # print("kl shape: ", kl.shape)

        # err = im - im_hat
        # serr = torch.square(err)
        # sse = torch.sum(serr)
        mse = torch.nn.MSELoss()(im, im_hat)
        self.mse = nn.Parameter(mse,requires_grad=False)
        self.kl = nn.Parameter(kl,requires_grad=False)
        # print("mse: ", mse)
        # print("kl: ", kl)
        # self.logger.info(f"MSE: {mse}; KL: {kl*self.alpha}")
        FE_simple = mse + self.alpha * kl
        # loss = self.alpha*torch.nn.MSELoss()(im_hat, im) + (1-self.alpha)*FE_simple
        #print('FE, mse:', mse)
        #print('FE, kl:', kl)
        # return mse
        return FE_simple


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
