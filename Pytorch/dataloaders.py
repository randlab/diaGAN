#! /usr/bin/env python3
# coding: utf-8

import argparse
import os
from gan import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from imageio import imread, imsave
import random
from mpstool.img import *

class Dataset2Cuts(Dataset):

    def __init__(self, img_x : str, img_y : str, epoch_size: int, batch_size : int, size : tuple, transform = None, binarize=False):
        self.img_x = torch.Tensor(imread(img_x)) / 255.
        self.img_y = torch.Tensor(imread(img_y)) / 255.

        if len(self.img_x.size()) > 2 :
            self.img_x = self.img_x[...,0]

        if len(self.img_y.size()) > 2 :
            self.img_y = self.img_y[...,0]

        self.shape_x = self.img_x.size()
        self.shape_y = self.img_y.size()

        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.sample_size = size
        self.transform = transform

    def get(self):
        sample = torch.zeros((self.batch_size, 2,) + self.sample_size[:2] )
        for n in range(self.batch_size):

            rxx = random.randint(0, self.shape_x[0] - self.sample_size[1]-1)
            rxy = random.randint(0, self.shape_x[1] - self.sample_size[2]-1)

            ryx = random.randint(0, self.shape_y[0] - self.sample_size[0]-1)
            ryy = random.randint(0, self.shape_y[1] - self.sample_size[2]-1)

            sample[n,0,...] = self.img_x[rxx:rxx + self.sample_size[1], rxy:rxy+self.sample_size[2]]
            sample[n,1,...] = self.img_y[ryx:ryx + self.sample_size[0], ryy:ryy+self.sample_size[2]]

        if self.transform:
            sample = self.transform(sample)
                    
        output = torch.Tensor(sample)
        return output

    def __getitem__(self, idx):
       return self.get()

class Dataset3Cuts(Dataset):

    def __init__(self, img_x : str, img_y : str, img_z : str, epoch_size: int, batch_size : int, size : tuple, transform = None, binarize= False):
        self.img_x = torch.Tensor(imread(img_x)) / 255.
        self.img_y = torch.Tensor(imread(img_y)) / 255.
        self.img_z = torch.Tensor(imread(img_z)) / 255.

        if len(self.img_x.size()) > 2 :
            self.img_x = self.img_x[...,0]

        if len(self.img_y.size()) > 2 :
            self.img_y = self.img_y[...,0]

        if len(self.img_z.size()) > 2 :
            self.img_z = self.img_z[...,0]

        self.shape_x = self.img_x.size()
        self.shape_y = self.img_y.size()
        self.shape_z = self.img_z.size()

        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.sample_size = size
        self.transform = transform

    def get(self):
        sample = torch.zeros((self.batch_size, 3,) + self.sample_size[:2] )
        for n in range(self.batch_size):

            rxx = random.randint(0, self.shape_x[0] - self.sample_size[1]-1)
            rxy = random.randint(0, self.shape_x[1] - self.sample_size[2]-1)

            ryx = random.randint(0, self.shape_y[0] - self.sample_size[0]-1)
            ryy = random.randint(0, self.shape_y[1] - self.sample_size[2]-1)

            rzx = random.randint(0, self.shape_z[1] - self.sample_size[0]-1)
            rzy = random.randint(0, self.shape_z[1] - self.sample_size[1]-1)

            sample[n,0,...] = self.img_x[rxx:rxx + self.sample_size[1], rxy:rxy+self.sample_size[2]]
            sample[n,1,...] = self.img_y[ryx:ryx + self.sample_size[0], ryy:ryy+self.sample_size[2]]
            sample[n,2,...] = self.img_z[rzx:rzx + self.sample_size[0], rzy:rzy+self.sample_size[1]]

        if self.transform:
            sample = self.transform(sample)
                    
        output = torch.Tensor(sample)
        return output

    def __getitem__(self, idx):
       return self.get()


class Dataset3DasCuts(Dataset):
    def __init__(self, img_paths : list, epoch_size: int, batch_size : int, size : tuple, transform = None, binarize=False):
        
        self.imgs = []
        for path in img_paths:
            if ".gslib" in path:
                self.imgs.append(Image.fromGslib(path))
            elif ".vox" in path:
                self.imgs.append(Image.fromVox(path))
            else:
                raise Exception("Image path '{}' could not be opened by mpstools".format(path))

        for i, img in enumerate(self.imgs):
            if binarize:
                img.threshold([1], [0,255])
            self.imgs[i] = torch.Tensor(img.asArray())
            

        self.shape = [img.shape for img in self.imgs]
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.sample_size = size
        self.transform = transform

    def get(self):

        sample = torch.zeros((self.batch_size, 3,) + self.sample_size[:2] )
        sx,sy,sz = self.sample_size        
        for n in range(self.batch_size):

            ind_img = random.randint(0, len(self.imgs)-1)
            simg = self.imgs[ind_img]

            rx = random.randint(0, self.shape[ind_img][0] - sx)
            cx = random.randint(0, sx-1)
            
            ry = random.randint(0, self.shape[ind_img][1] - sy)
            cy = random.randint(0, sy-1)
        
            rz = random.randint(0, self.shape[ind_img][2] - sz)
            cz = random.randint(0, sz-1)

            sample[n,0,...] = simg[rx + cx, ry:ry+sy, rz:rz+sz]
            sample[n,1,...] = simg[rx:rx+sx, ry+cy, rz:rz+sz]
            sample[n,2,...] = simg[rx:rx+sx, ry:ry+sy, rz+cz]

        if self.transform:
            sample = self.transform(sample)        
        output = torch.Tensor(sample)
        return output

    def __getitem__(self, idx):
       return self.get()