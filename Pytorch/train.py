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
from dataloaders import *
from mpstool.img import Image
import numpy as np
import random
from tqdm import tqdm

def extract_3cuts(tensor):
    bs,_,sx,sy,sz = tensor.size()
    rx = torch.randint(0, sx-1, (1,))[0]
    ry = torch.randint(0, sy-1, (1,))[0]
    rz = torch.randint(0, sz-1, (1,))[0]

    return torch.cat([
        tensor[:, :, rx, :, :],
        tensor[:, :, :, ry, :],
        tensor[:, :, :, :, rz]
    ], dim=1) 


def gradient_penalty(critic, args, device, real_data, fake_data):
    alpha = torch.FloatTensor(args.batch_size, 1, 1, 1).uniform_(0.,1.)
    alpha = alpha.expand(args.batch_size, real_data.size(1), real_data.size(2), real_data.size(3))
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    D_interpolated = critic(interpolates)

    gradients = torch.autograd.grad(
        outputs=D_interpolated,
        inputs=interpolates,
        grad_outputs=torch.ones(D_interpolated.size()).to(device),
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lmbd
    return gradient_penalty


def train_one_epoch(epoch, generator, opt_gen, critic, opt_crit, args, device, data):
    for i_batch in tqdm(range(args.epoch_size)):

        # ------ Train critic
        generator.eval()
        critic.train()
        for p in critic.parameters():
            p.requires_grad = True

        for _ in range(args.n_critic):
            opt_crit.zero_grad()

            # train with real
            real_data = data.get()
            real_data_v = torch.autograd.Variable(real_data).to(device)
            crit_real = critic(real_data_v)
            crit_real = crit_real.mean()

            # train with fake
            noise = get_noise((args.batch_size,generator.noise_size)).to(device)
            with torch.no_grad():
                noise_v = torch.autograd.Variable(noise)
                fake_data = generator(noise_v).data
                fake_data = extract_3cuts(fake_data)
            fake_data_v = torch.autograd.Variable(fake_data)
            crit_fake = critic(fake_data_v)
            crit_fake = crit_fake.mean()

            # train with gradient penalty
            gp = gradient_penalty(critic, args, device, real_data_v.data, fake_data_v.data)

            crit_loss = crit_fake - crit_real + gp
            crit_loss.backward()
            opt_crit.step()

        # ------ Train generator
        generator.train()
        critic.eval()
        for p in critic.parameters():
            p.requires_grad = False
        opt_gen.zero_grad()

        noise = get_noise((args.batch_size,generator.noise_size)).to(device)
        noise_v = torch.autograd.Variable(noise)
        gen_out = generator(noise_v)
        crit_in = extract_3cuts(gen_out)
        score = critic(crit_in)
        score = score.mean()
        gen_loss = -score
        gen_loss.backward()
        opt_gen.step()


def generate(epoch, generator, N, args, device):
    for i in range(N):
        output = generator.generate(1, device).cpu().detach().numpy()
        output = np.squeeze(output)
        output = output * 255
        output = Image.fromArray(output)
        output.exportAsVox("output/epoch{}_{}.vox".format(epoch, i))


if __name__=="__main__":

    # Training settings
    parser = argparse.ArgumentParser(description='ArchiSearch Guillaume IMT Atlantique')

    parser.add_argument('-dx', type=str, default=None)
    parser.add_argument('-dy', type=str, default=None)
    parser.add_argument('-dz', type=str, default=None)

    parser.add_argument("-dataset", type=str, default=None)

    parser.add_argument("-checkpoint-freq", type=int, default=1)

    # Optionnal Arguments
    parser.add_argument('--lr', type=float, default=1E-3)

    parser.add_argument('--batch-size', type=int, default=10)

    parser.add_argument('--epoch-size', type=int, default=1000)

    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--lmbd', type=float, default=10, help="gradient penalty weight")

    parser.add_argument('--n-critic', type=int, default=5)

    parser.add_argument("--seed", type=int, default=random.randint(0,1000000))

    args = parser.parse_args()
    assert args.dx is not None or args.dataset is not None
    print(args)

    try:
        os.mkdir("output")
    except:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    generator = Generator(256)
    critic = Critic()

    generator, critic = generator.to(device), critic.to(device)

    optimizer_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optimizer_crit = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

    if args.dx is not None:
        data = Dataset3Cuts(args.dx, args.dy, args.dz, args.n_critic*args.epoch_size, args.batch_size, (64, 64, 64))
    else:
        data = Dataset3DasCuts(args.dataset, args.n_critic*args.epoch_size, args.batch_size, (64, 64, 64))

    for epoch in range(1, args.epochs+1):
        train_one_epoch(epoch, generator, optimizer_gen, critic, optimizer_crit, args, device, data)
        generate(epoch, generator, 3, args, device)
        if epoch%args.checkpoint_freq==0:
            torch.save(generator.state_dict(), "output/gen_e{}.model".format(epoch))

    
