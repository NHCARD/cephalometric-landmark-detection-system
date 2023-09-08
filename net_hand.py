import glob

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from mytransforms import *
from mytransforms import mytransforms
from skimage.filters import threshold_otsu
from skimage import feature
from skimage.color import rgb2gray
from numpy import matlib
import cv2
import os, sys
import numpy as np
from numpy import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import time


def gray_to_rgb(gray):
    h, w = gray.shape
    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = gray;
    rgb[:, :, 1] = gray;
    rgb[:, :, 2] = gray;
    return rgb


class dataload(Dataset):
    def __init__(self, path='train', H=600, W=480, pow_n=3, aug=True, mode='img'):

        self.mode = mode
        if mode == 'img':
            self.path = path
            self.data_num = 1
        elif mode == 'dir':
            self.path = glob.glob(path + '/*.png')
            self.data_num = len(self.path)
        # self.mask_num = int(len(self.dinfo.classes))  # 20
        # # print(self.mask_num)
        # # print(self.mask_num)
        # # print(self.data_num)
        # self.path_mtx = np.array(self.dinfo.samples)[:, :1].reshape(self.mask_num,
        #                                                             self.data_num)  ## all data path loading  [ masks  20 , samples 150]
        # self.images = [Image.open(path) for path in self.path_mtx.reshape(-1)]  # all image loading
        # self.path1D = self.path_mtx.reshape(-1)  # all image path list
        # print(self.path_mtx)

        self.aug = aug
        self.pow_n = pow_n
        self.task = path
        self.H = H
        self.W = W
        # augmenation of img and masks
        self.mask_trans = transforms.Compose([transforms.Resize((self.H, self.W)),
                                              transforms.Grayscale(1),
                                              # mytransforms.Affine(0,
                                              #                     translate=[0, 0],
                                              #                     scale=1,
                                              #                     fillcolor=0),
                                              transforms.ToTensor(),

                                              ])
        self.norm = mytransforms.Compose([transforms.Normalize((0.5,), (0.5,))])
        self.col_trans = transforms.Compose([transforms.ColorJitter(brightness=random.random(),
                                                                    # contrast=random.random(),
                                                                    # saturation=random.random(),
                                                                    # hue=random.random() / 2
                                                                    ),
                                             ])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        # mask = torch.empty(self.mask_num, self.H, self.W, dtype=torch.float)  # 150 * H * W
        # if self.aug == True:
        #     self.mask_trans.transforms[2].degrees = random.randrange(-25, 25)
        #     self.mask_trans.transforms[2].translate = [random.uniform(0, 0.05), random.uniform(0, 0.05)]
        #     self.mask_trans.transforms[2].scale = random.uniform(0.9, 1.1)

        # for k in range(0, self.mask_num):
        #     X = Image.open(self.path_mtx[k, idx])
        #     if k == 0 and self.aug == True: X = self.col_trans(X)
        #     mask[k] = self.mask_trans(X)

        # input, heat = self.norm(mask[0:1]), mask[1:38]
        # heat = torch.pow(heat, self.pow_n)
        # heat = heat / heat.max()
        if self.mode == 'img':
            input = Image.open(self.path)
        elif self.mode == 'dir':
            input = Image.open(self.path[idx])
        input = self.mask_trans(input)
        input = self.norm(input)
        img_name = self.path
        img_size = input.size()
        # plt.imshow(input[0], cmap='gray');
        # plt.show()
        # plt.imshow(heat[0], cmap='gray');
        # plt.show()
        # print("idx :", idx, "path ", self.task)

        return [input, img_size, img_name]


# UNET parts
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


## Original
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        factor = 2
        # print(factor)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


###############

class AttDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, att):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            att,
        )

    def forward(self, x):
        return self.double_conv(x)


class AttDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, att):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            AttDoubleConv(in_channels, out_channels, att)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, att):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = AttDoubleConv(in_channels, out_channels, att)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


## ArgPool

H = 800;
W = 640


## SENET
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
        )

    def forward(self, x):
        b, c, *_ = x.size()
        avg_pool = torch.mean(x.view(b, c, -1), dim=-1)
        scale = torch.sigmoid(self.mlp(avg_pool))

        scale = scale.view((b, c) + (1,) * len(_))  # (b, c, 1, 1, 1)
        scale = scale.expand_as(x)

        return x * scale


class SELayer(nn.Module):
    def __init__(self, channel=512, reduction=8):  # 16
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SENet(nn.Module):
    def __init__(self, n_channels, n_classes, reduce=16, bilinear=False):
        super(SENet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2
        print(factor, 'reduce', reduce)

        self.inc = AttDoubleConv(n_channels, 64, att=SELayer(channel=64, reduction=reduce))
        self.down1 = AttDown(64, 128, att=SELayer(channel=128, reduction=reduce))
        self.down2 = AttDown(128, 256, att=SELayer(channel=256, reduction=reduce))
        self.down3 = AttDown(256, 512, att=SELayer(channel=512, reduction=reduce))
        self.down4 = AttDown(512, 1024 // factor, att=SELayer(channel=512, reduction=reduce))
        self.up1 = AttUp(1024, 512 // factor, att=SELayer(channel=256, reduction=reduce))
        self.up2 = AttUp(512, 256 // factor, att=SELayer(channel=128, reduction=reduce))
        self.up3 = AttUp(256, 128 // factor, att=SELayer(channel=64, reduction=reduce))
        self.up4 = AttUp(128, 64, att=SELayer(channel=64, reduction=reduce))
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


## CBAM
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBAMNet(nn.Module):
    def __init__(self, n_channels, n_classes, reduce=16, bilinear=False):
        super(CBAMNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2
        print(factor, 'reduce', reduce)

        self.inc = AttDoubleConv(n_channels, 64, att=CBAM(64, reduction_ratio=reduce))
        self.down1 = AttDown(64, 128, att=CBAM(128, reduction_ratio=reduce))
        self.down2 = AttDown(128, 256, att=CBAM(256, reduction_ratio=reduce))
        self.down3 = AttDown(256, 512, att=CBAM(512, reduction_ratio=reduce))
        self.down4 = AttDown(512, 1024 // factor, att=CBAM(512, reduction_ratio=reduce))
        self.up1 = AttUp(1024, 512 // factor, att=CBAM(256, reduction_ratio=reduce))
        self.up2 = AttUp(512, 256 // factor, att=CBAM(128, reduction_ratio=reduce))
        self.up3 = AttUp(256, 128 // factor, att=CBAM(64, reduction_ratio=reduce))
        self.up4 = AttUp(128, 64, att=CBAM(64, reduction_ratio=reduce))
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


## ECA
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:channel: Number of channels of the input feature map// k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=7):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECANet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ECANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        factor = 2

        self.inc = AttDoubleConv(n_channels, 64, att=eca_layer(channel=64))
        self.down1 = AttDown(64, 128, att=eca_layer(channel=128))
        self.down2 = AttDown(128, 256, att=eca_layer(channel=256))
        self.down3 = AttDown(256, 512, att=eca_layer(channel=512))
        self.down4 = AttDown(512, 1024 // factor, att=eca_layer(channel=512))
        self.up1 = AttUp(1024, 512 // factor, att=eca_layer(channel=256))
        self.up2 = AttUp(512, 256 // factor, att=eca_layer(channel=128))
        self.up3 = AttUp(256, 128 // factor, att=eca_layer(channel=64))
        self.up4 = AttUp(128, 64, att=eca_layer(channel=64))
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


## GSOP
class GSOP1_layer(nn.Module):
    def __init__(self, in_c=512, dr_c=64, maxpool=False):  # 16
        super(GSOP1_layer, self).__init__()

        if maxpool:
            print("max pool", maxpool)
            self.relu_normal = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.MaxPool2d(maxpool)
            )
        else:
            self.relu_normal = nn.ReLU(inplace=False)

        self.relu = nn.ReLU(inplace=True)
        self.conv_for_DR = nn.Conv2d(in_c, dr_c, kernel_size=1, stride=1, bias=True)
        self.bn_for_DR = nn.BatchNorm2d(dr_c)

        self.row_conv_group = nn.Conv2d(dr_c, 4 * dr_c, kernel_size=(dr_c, 1),
                                        groups=dr_c, bias=True)
        self.fc_adapt_channels = nn.Conv2d(4 * dr_c, in_c, kernel_size=1, groups=1, bias=True)
        self.row_bn = nn.BatchNorm2d(dr_c)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # NxCxHxW
        y = self.relu_normal(x)
        y = self.conv_for_DR(y)
        y = self.bn_for_DR(y)
        y = self.relu(y)

        y = CovpoolLayer(y)  # Nxdxd
        y = y.view(y.size(0), y.size(1), y.size(2), 1).contiguous()  # Nxdxdx1
        y = self.row_bn(y)
        y = self.row_conv_group(y)  # Nx512x1x1
        y = self.fc_adapt_channels(y)  # NxCx1x1
        y = self.sigmoid(y)  # NxCx1x1
        return x * y


class GSOPNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(GSOPNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        factor = 2

        self.inc = DoubleConv(n_channels, 64)  # 800
        self.down1 = AttDown(64, 128, att=GSOP1_layer(128, 64, maxpool=4))  # 400
        self.down2 = AttDown(128, 256, att=GSOP1_layer(256, 64))  # 200
        self.down3 = AttDown(256, 512, att=GSOP1_layer(512, 64))  # 100
        self.down4 = AttDown(512, 1024 // factor, att=GSOP1_layer(512, 64))  # 50
        self.up1 = AttUp(1024, 512 // factor, att=GSOP1_layer(256, 64))
        self.up2 = AttUp(512, 256 // factor, att=GSOP1_layer(128, 64))
        self.up3 = AttUp(256, 128 // factor, att=GSOP1_layer(64, 64, maxpool=4))
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class context_layer(nn.Module):
    def __init__(self, in_c=512, dr_c=64, maxpool=False):  # 16
        super(context_layer, self).__init__()

        if maxpool:
            print("max pool", maxpool)
            self.relu_normal = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.MaxPool2d(maxpool)
            )
        else:
            self.relu_normal = nn.ReLU(inplace=False)

        self.relu = nn.ReLU(inplace=True)
        self.conv_for_DR = nn.Conv2d(in_c, dr_c, kernel_size=1, stride=1, bias=True)
        self.bn_for_DR = nn.InstanceNorm2d(dr_c, affine=True)

        self.row_conv_group = nn.Conv2d(dr_c, 4 * dr_c, kernel_size=(dr_c, 1),
                                        groups=dr_c, bias=True)
        self.fc_adapt_channels = nn.Conv2d(4 * dr_c, in_c, kernel_size=1, groups=1, bias=True)
        self.row_bn = nn.InstanceNorm2d(dr_c, affine=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # NxCxHxW
        y = self.relu_normal(x)
        y = self.conv_for_DR(y)
        y = self.bn_for_DR(y)
        y = self.relu(y)

        y = CovpoolLayer(y)  # Nxdxd
        y = y.view(y.size(0), y.size(1), y.size(2), 1).contiguous()  # Nxdxdx1
        y = self.row_bn(y)
        y = self.row_conv_group(y)  # Nx512x1x1
        y = self.fc_adapt_channels(y)  # NxCx1x1
        y = self.sigmoid(y)  # NxCx1x1
        return x * y


class contextNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(contextNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        factor = 2

        self.inc = DoubleConv(n_channels, 64)  # 800
        self.down1 = AttDown(64, 128, att=context_layer(128, 64, maxpool=4))  # 400
        self.down2 = AttDown(128, 256, att=context_layer(256, 64))  # 200
        self.down3 = AttDown(256, 512, att=context_layer(512, 64))  # 100
        self.down4 = AttDown(512, 1024 // factor, att=context_layer(512, 64))  # 50
        self.up1 = AttUp(1024, 512 // factor, att=context_layer(256, 64))
        self.up2 = AttUp(512, 256 // factor, att=context_layer(128, 64))
        self.up3 = AttUp(256, 128 // factor, att=context_layer(64, 64, maxpool=4))
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


## 3D Gsop
import torch
import torch.nn as nn
from torch.autograd import Function


class Covpool(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        z = x.data.shape[4]
        M = h * w * z
        x = x.reshape(batchSize, dim, M)
        I_hat = (-1. / M / M) * torch.ones(M, M, device=x.device) + (1. / M) * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        ctx.save_for_backward(input, I_hat)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        z = x.data.shape[4]
        M = h * w * z
        x = x.reshape(batchSize, dim, M)
        grad_input = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize, dim, h, w, z)
        return grad_input


class GSOP3d_layer(nn.Module):
    def __init__(self, in_c=10, dr_c=10):  # 16
        super(GSOP3d_layer, self).__init__()

        self.relu_normal = nn.ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_for_DR = nn.Conv2d(in_c, dr_c, kernel_size=1, stride=1, bias=True)
        self.conv_for_DR_3d = nn.Conv3d(in_c, dr_c, kernel_size=1, stride=1, bias=True)

        self.row_conv_group = nn.Conv2d(dr_c, 8 * dr_c, kernel_size=(dr_c, 1),
                                        groups=dr_c, bias=True)
        self.fc_adapt_channels = nn.Conv2d(8 * dr_c, in_c, kernel_size=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # BxCxHxWxZ
        y = self.relu_normal(x)
        y = self.conv_for_DR_3d(y)  # 채널 줄이기
        # out = self.bn_for_DR(out)  # 원래 코드 배치 정규화

        y = self.relu(y)
        y = Covpool.apply(y)  # out : BxCxC
        y = y.view(y.size(0), y.size(1), y.size(2), 1).contiguous()  # BxCxCx1
        # out = self.row_bn(out) # 원래 코드 배치 정규화

        y = self.row_conv_group(y)  # Bx( C* hp )x1x1
        y = self.fc_adapt_channels(y)  # BxCx1x1
        y = y.unsqueeze(4)  # BxCx1x1x1
        y = self.sigmoid(y)  # NxCx1x1
        return x * y


dd = GSOP3d_layer(in_c=20, dr_c=10)
x = torch.ones((1, 20, 10, 10, 10))
out = dd(x)
# print(out.shape)
