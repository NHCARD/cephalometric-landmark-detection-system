import glob

from torchvision import transforms
from numpy import *
from PIL import Image
from torch.utils.data import Dataset

from mytransforms import mytransforms


class dataload(Dataset):
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
