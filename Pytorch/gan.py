import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_noise(size):
    return torch.Tensor(np.random.normal(0., 0.5, size).astype(np.float32))

class Generator(nn.Module):

    def __init__(self, noise_size):
        super(Generator,self).__init__()
        self.noise_size = noise_size
        self.embed_shape = (1, 16, 16, 16)
        self.embed_size = np.prod(self.embed_shape)

        self.upscale = lambda x : F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)

        self.LinEmb = nn.Sequential(
            nn.Linear(self.noise_size, self.embed_size),
            nn.ReLU6(),
        )
        #self.LinEmb = nn.DataParallel(self.LinEmb)

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU6(),
        )
        #self.conv1 = nn.DataParallel(self.conv1)

        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU6(),
        )
        #self.conv2 = nn.DataParallel(self.conv2)

        self.conv3 = nn.Sequential(   
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU6(),
        )
        #self.conv3 = nn.DataParallel(self.conv3)
        
        self.conv4 = nn.Sequential(   
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU6(),
        )
        #self.conv4 = nn.DataParallel(self.conv4)

        self.conv5 = nn.Sequential(
            nn.Conv3d(32, 1,  kernel_size=3, padding=1),
            nn.ReLU6()
        )
        #self.conv5 = nn.DataParallel(self.conv5)
    

    def forward(self, x):

        x = self.LinEmb(x)
        x = x.view((x.size(0),) + self.embed_shape)
    
        x = self.conv1(x)
        x = self.upscale(x)

        x = self.conv2(x)
        x = self.upscale(x)

        x = self.conv3(x)

        x = self.conv4(x)
        
        x = self.conv5(x)
        return x/6 # for numbers in interval [0;1] and not [0;6]        


    def generate(self, batch_size=1, device="cuda"):
        if not torch.cuda.is_available():
            device = "cpu"
        return self.forward(get_noise((batch_size,self.noise_size)).to(device))


class Critic(nn.Module):

    def __init__(self):
        super(Critic,self).__init__()
    
        self.Lconv1 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        #self.Lconv1 = nn.DataParallel(self.Lconv1)

        self.Lconv2 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        #self.Lconv2 = nn.DataParallel(self.Lconv2)

        self.Lconv3 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        #self.Lconv3 = nn.DataParallel(self.Lconv3)

        self.Lconv4 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        #self.Lconv4 = nn.DataParallel(self.Lconv4)

        self.Lconv5 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
                nn.ReLU()
        )
        #self.Lconv5 = nn.DataParallel(self.Lconv5)

        self.Loutput = nn.Linear(128,1)
        self.Loutput = nn.DataParallel(self.Loutput)
        
    def forward(self, x):
        
        x = self.Lconv1(x)
        x = self.Lconv2(x)
        x = self.Lconv3(x)
        x = self.Lconv4(x)
        x = self.Lconv5(x)

        size = x.size()[-1]
        out = F.max_pool2d(x, kernel_size=size)[...,0,0]
        out = self.Loutput(out)

        return out