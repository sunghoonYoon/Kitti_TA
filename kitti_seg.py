"""
This code is from https://github.com/NVIDIA/semantic-segmentation
"""


"""
Custom Norm wrappers to enable sync BN, regular BN and for weight initialization
"""
import torch.nn as nn
import numpy as np
# from config import cfg

# from apex import amp

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = torch.nn.BatchNorm2d #getattr(cfg.MODEL, 'BNFUNC')
    normalization_layer = layer(in_channels)
    return normalization_layer


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# @amp.float_function
def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


"""
# Code adapted from:
# https://github.com/mapillary/inplace_abn/
#
# BSD 3-Clause License
#
# Copyright (c) 2017, mapillary
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import logging
import sys
from collections import OrderedDict
from functools import partial
import torch.nn as nn
import torch
# import network.mynn as mynn

def bnrelu(channels):
    """
    Single Layer BN and Relui
    """
    return nn.Sequential(Norm2d(channels),
                         nn.ReLU(inplace=True))

class GlobalAvgPool2d(nn.Module):
    """
    Global average pooling over the input's spatial dimensions
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        logging.info("Global Average Pooling Initialized")

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class IdentityResidualBlock(nn.Module):
    """
    Identity Residual Block for WideResnet
    """
    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=bnrelu,
                 dropout=None,
                 dist_bn=False
                 ):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps.
            Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions,
            otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups.
            This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        dist_bn: Boolean
            A variable to enable or disable use of distributed BN
        """
        super(IdentityResidualBlock, self).__init__()
        self.dist_bn = dist_bn

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn


        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels,
                                    channels[0],
                                    3,
                                    stride=stride,
                                    padding=dilation,
                                    bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1],
                                    3,
                                    stride=1,
                                    padding=dilation,
                                    bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1",
                 nn.Conv2d(in_channels,
                           channels[0],
                           1,
                           stride=stride,
                           padding=0,
                           bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0],
                                    channels[1],
                                    3, stride=1,
                                    padding=dilation, bias=False,
                                    groups=groups,
                                    dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2],
                                    1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        """
        This is the standard forward function for non-distributed batch norm
        """
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)
        return out




class WiderResNet(nn.Module):
    """
    WideResnet Global Module for Initialization
    """
    def __init__(self,
                 structure,
                 norm_act=bnrelu,
                 classes=0
                 ):
        """Wider ResNet with pre-activation (identity mapping) blocks

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and \
            a fully-connected layer with `classes` outputs at the end
            of the network.
        """
        super(WiderResNet, self).__init__()
        self.structure = structure

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024),
                    (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels, channels[mod_id],
                                          norm_act=norm_act)
                ))

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id <= 4:
                self.add_module("pool%d" %
                                (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(self.pool4(out))
        out = self.mod5(self.pool5(out))
        out = self.mod6(self.pool6(out))
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            out = self.classifier(out)

        return out


class WiderResNetA2(nn.Module):
    """
    Wider ResNet with pre-activation (identity mapping) blocks

    This variant uses down-sampling by max-pooling in the first two blocks and
     by strided convolution in the others.

    Parameters
    ----------
    structure : list of int
        Number of residual blocks in each of the six modules of the network.
    norm_act : callable
        Function to create normalization / activation Module.
    classes : int
        If not `0` also include global average pooling and a fully-connected layer
        with `classes` outputs at the end
        of the network.
    dilation : bool
        If `True` apply dilation to the last three modules and change the
        down-sampling factor from 32 to 8.
    """
    def __init__(self,
                 structure,
                 norm_act=bnrelu,
                 classes=0,
                 dilation=False,
                 dist_bn=False
                 ):
        super(WiderResNetA2, self).__init__()
        self.dist_bn = dist_bn

        # If using distributed batch norm, use the encoding.nn as oppose to torch.nn


        nn.Dropout = nn.Dropout2d
        norm_act = bnrelu
        self.structure = structure
        self.dilation = dilation

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = torch.nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                    (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout, p=0.5)
                else:
                    drop = None

                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels,
                                          channels[mod_id], norm_act=norm_act,
                                          stride=stride, dilation=dil,
                                          dropout=drop, dist_bn=self.dist_bn)
                ))

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id < 2:
                self.add_module("pool%d" %
                                (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            return self.classifier(out)
        return out


_NETS = {
    "16": {"structure": [1, 1, 1, 1, 1, 1]},
    "20": {"structure": [1, 1, 1, 3, 1, 1]},
    "38": {"structure": [3, 3, 6, 3, 1, 1]},
}

__all__ = []
for name, params in _NETS.items():
    net_name = "wider_resnet" + name
    setattr(sys.modules[__name__], net_name, partial(WiderResNet, **params))
    __all__.append(net_name)
for name, params in _NETS.items():
    net_name = "wider_resnet" + name + "_a2"
    setattr(sys.modules[__name__], net_name, partial(WiderResNetA2, **params))
    __all__.append(net_name)

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class DeepWV3Plus(nn.Module):
    """
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    """

    def __init__(self, num_classes, trunk='WideResnet38', criterion=None):

        super(DeepWV3Plus, self).__init__()
        self.criterion = criterion
        logging.info("Trunk: %s", trunk)
        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        # TODO: Should this be even here ?
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        try:
            checkpoint = torch.load('./pretrained_models/wider_resnet38.pth.tar', map_location='cpu')
            wide_resnet.load_state_dict(checkpoint['state_dict'])
            del checkpoint
        except:
            print("=====================Could not load ImageNet weights=======================")
            print("Please download the ImageNet weights of WideResNet38 in our repo to ./pretrained_models.")

        wide_resnet = wide_resnet.module

        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet

        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final)

    def forward(self, inp, gts=None):

        x_size = inp.size()
        x = self.mod1(inp)
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(m2)
        dec0_up = Upsample(dec0_up, m2.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final(dec0)
        out = Upsample(dec1, x_size[2:])

        if self.training:
            return self.criterion(out, gts)

        return out

def decode_segmap(temp):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],#road
        [244, 35, 232],#sidewalk
        [70, 70, 70],#building
        [102, 102, 156],#wall
        [190, 153, 153],#fence
        [153, 153, 153],#pole
        [250, 170, 30],#traffic light
        [220, 220, 0],#trafiic sign
        [107, 142, 35],  # vegetation dark green
        [152, 251, 152],  # terrain bright green
        [0, 130, 180],#sky
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def denorm(img):
    # ImageNet statistics
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    tf_denorm = transforms.Normalize(mean = [-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
                                     std = [1 / std_img[0], 1 / std_img[1], 1 / std_img[2]])

    return tf_denorm(img)