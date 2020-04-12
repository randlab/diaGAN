#! /usr/bin/env python3
# coding: utf-8

import argparse
import os
from gan import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from imageio import imread, imsave
from dataloaders import *
from mpstool.img import Image
import numpy as np
import random
import tqdm


def generate(generator, N, args, device):
    for i in tqdm.trange(N):
        output = generator.generate(1, device).cpu().detach().numpy()
        output = np.squeeze(output)
        output = (output * 255).astype(np.uint8)
        output = Image.fromArray(output)
        if args.binarize:
            output.threshold([127], [0,255])
        output.exportAsVox("output/{}_{}.vox".format(args.name, i))


if __name__=="__main__":

    # Training settings
    parser = argparse.ArgumentParser(description='DiAGAN generation script')

    parser.add_argument('-model', "--model", type=str, required=True)
    parser.add_argument("-n", "--n", type=int, default=1)
    parser.add_argument("-name", "--name", type=str, default="output")
    parser.add_argument("-binarize", "--binarize", action="store_true")
    args = parser.parse_args()

    try:
        os.mkdir("output")
    except:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = Generator(256)
    generator = generator.to(device)
    generator.load_state_dict(torch.load(args.model))
    
    generate(generator, args.n, args, device)
    print("Successfully generated {} samples in the output/ folder".format(args.n))
    
