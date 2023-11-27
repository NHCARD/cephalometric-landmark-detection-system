import glob

import torch.nn as nn
import torch
from torchvision import transforms, datasets, models
from mytransforms import mytransforms
import numpy as np
from numpy import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import cv2


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
        img_size = input.size()
        # plt.imshow(input[0], cmap='gray');
        # plt.show()
        # plt.imshow(heat[0], cmap='gray');
        # plt.show()
        # print("idx :", idx, "path ", self.task)

        return input


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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
