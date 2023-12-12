import glob

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

from mytransforms import mytransforms
import numpy as np
import torchvision
import random
import torch

class dataload_valid(Dataset):
    def __init__(self, path='train', H=600, W=480, pow_n=3, aug=True, mode='img'):
        self.H = H
        self.W = W
        self.pow_n = pow_n
        self.aug = aug
        self.mode = mode

        if mode == 'img':
            self.path = path
            self.data_num = 1
        elif mode == 'dir':
            self.path = glob.glob(path + '/*.png')
            self.data_num = len(self.path)

        self.mask_trans = transforms.Compose([transforms.Resize((self.H, self.W)),
                                              transforms.Grayscale(1),
                                              transforms.ToTensor()])
        self.norm = mytransforms.Compose([transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.mode == 'img':
            input = Image.open(self.path)
        if self.mode == 'dir':
            input = Image.open(self.path[idx])
        input = self.mask_trans(input)
        input = self.norm(input)

        return input


class dataload_train(Dataset):
    def __init__(self,  path='train', H=600, W=480, pow_n=3, aug=True):

        self.dinfo = torchvision.datasets.ImageFolder(root=path)
        self.mask_num = int(len(self.dinfo.classes))## 20
        self.data_num = int(len(self.dinfo.targets) / self.mask_num) ##150
        self.path_mtx = np.array(self.dinfo.samples)[:, :1].reshape(self.mask_num,
                                                                    self.data_num)  ## all data path loading  [ masks  20 , samples 150]
        self.aug=aug
        self.pow_n = pow_n
        self.task = path
        self.H = H
        self.W = W

        self.mask_trans = transforms.Compose([transforms.Resize((self.H, self.W)),
                                              transforms.Grayscale(1),
                                              mytransforms.Affine(0, translate=[0, 0], scale=1, fillcolor=0),
                                              transforms.ToTensor()])

        self.norm = mytransforms.Compose([transforms.Normalize((0.5,), (0.5,))])
        self.col_trans = transforms.Compose([transforms.ColorJitter(brightness=random.random())])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        mask = torch.empty(self.mask_num, self.H, self.W, dtype=torch.float)  # 150 * H * W
        if self.aug:
            self.mask_trans.transforms[2].degrees = random.randrange(-25, 25)
            self.mask_trans.transforms[2].translate = [random.uniform(0, 0.05), random.uniform(0, 0.05)]
            self.mask_trans.transforms[2].scale= random.uniform(0.9, 1.1)

        for k in range(0, self.mask_num):
            X = Image.open(self.path_mtx[k, idx])
            if k == 0 and self.aug == True: X = self.col_trans(X)
            mask[k] = self.mask_trans(X)

        input, heat = self.norm(mask[0:1]), mask[1:]
        heat = torch.pow(heat, self.pow_n)
        heat = heat / heat.max()

        return [input, heat]
