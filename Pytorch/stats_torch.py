import os
import numpy as np
import matplotlib.pyplot as plt
import mpstool
from mpstool.img import Image
from random import randint
import argparse
import torch
from gan import Generator
from tqdm import tqdm

GRAY = '#a0a0a0'
ALPHA = 0.3

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-fun", "--fun", type=str, required=True)
    parser.add_argument('-model', "--model", type=str, required=True)
    parser.add_argument('-ti', "--ti", type=str, required=True)
    parser.add_argument('-output', "--output", type=str, default="output.csv")
    parser.add_argument("-n-samples", "--n-samples", type=int, default=100)
    parser.add_argument("-n-ti", "--n-ti", type=int, default=100)

    args = parser.parse_args()
    model = Generator(256)
    if not torch.cuda.is_available():
        model = model.to("cpu")
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
    else:
        model = model.to("cuda")
        model.load_state_dict(torch.load(args.model))

    if ".vox" in args.ti:
        ti = Image.fromVox(args.ti)
        ti.threshold(thresholds=[1],values=[0,1])
        ti = ti.asArray()
    elif ".gslib" in args.ti:
        ti = Image.fromGslib(args.ti).asArray()

    if args.fun == "conn":
        stat_fun = mpstool.connectivity.get_function
    elif args.fun == "vario":
        stat_fun = mpstool.stats.variogram
    else:
        raise Exception("argument mode not recognized. Should be 'conn' or 'vario' ")

    # Read connectivity for the TI : min and max of values
    cX_ti = []
    cY_ti = []
    cZ_ti = []
    cX_ti_mean = {}
    cY_ti_mean = {}
    cZ_ti_mean = {}

    sample_size = model.generate().to("cpu").size()[2:]

    for n in tqdm(range(args.n_ti)):
        ix = randint(0, ti.shape[0]-sample_size[0]-1)
        iy = randint(0, ti.shape[1]-sample_size[1]-1)
        if ti.shape[2]-sample_size[2]-1>0:
            iz = randint(0, ti.shape[2]-sample_size[2]-1)
        else:
            iz = 0

        sample = ti[ix:ix+sample_size[0], iy:iy+sample_size[1], iz:iz+sample_size[2]]

        connX = stat_fun(sample, axis=1)
        cX_ti.append(connX)
        cX_ti_mean = { k: cX_ti_mean.get(k, 0) + connX.get(k, 0) for k in set(connX) }

        connY = stat_fun(sample, axis=0)
        cY_ti.append(connY)
        cY_ti_mean = { k: cY_ti_mean.get(k, 0) + connY.get(k, 0) for k in set(connY) }

        connZ = stat_fun(sample, axis=2)
        cZ_ti.append(connZ)
        cZ_ti_mean = { k: cZ_ti_mean.get(k, 0) + connZ.get(k, 0) for k in set(connZ) }
    
    cX_ti_mean = {k: cX_ti_mean.get(k,0)/args.n_ti for k in cX_ti_mean.keys()}
    cY_ti_mean = {k: cY_ti_mean.get(k,0)/args.n_ti for k in cY_ti_mean.keys()}
    cZ_ti_mean = {k: cZ_ti_mean.get(k,0)/args.n_ti for k in cZ_ti_mean.keys()}

    cX_ti_std = {}
    cY_ti_std = {}
    cZ_ti_std = {}
    
    for std_dict, data_dict in [(cX_ti_std, cX_ti), 
                                (cY_ti_std, cY_ti), 
                                (cZ_ti_std, cZ_ti)]:
        for smpl in data_dict:
            for facies in smpl:
                if facies not in std_dict:
                    std_dict[facies] = [ data_dict[smpl][facies] ]
                else:
                    std_dict[facies].append( data_dict[smpl][facies] )
    
    cX_ti_std = {k : np.std(cX_ti_std[k], axis=1) for k in cX_ti_std.keys() }
    cY_ti_std = {k : np.std(cY_ti_std[k], axis=1) for k in cY_ti_std.keys() }
    cZ_ti_std = {k : np.std(cZ_ti_std[k], axis=1) for k in cZ_ti_std.keys() }

    # min -> mean - std
    # max -> mean + std
    # clip at [0;1]
 
    cX_ti_min, cX_ti_max = {}, {}
    cY_ti_min, cY_ti_max = {}, {}
    cZ_ti_min, cZ_ti_max = {}, {}

    for std_dict, mean_dict, min_dict, max_dict in 
        [(cX_ti_std, cX_ti_mean, cX_ti_min, cX_ti_max), 
         (cY_ti_std, cY_ti_mean, cY_ti_min, cY_ti_max), 
         (cZ_ti_std, cZ_ti_mean, cZ_ti_min, cZ_ti_max)]:
        for k in mean_dict.keys():
            min_dict[k] = mean_dict[k] - std_dict[k]
            min_dict[k].clip(0,1)
            max_dict[k] = mean_dict[k] + std_dict[k]
            max_dict[k].clip(0,1)

    # Read connectivity for the realizations
    cX = []
    cY = []
    cZ = []
    cX_mean = {}
    cY_mean = {}
    cZ_mean = {}

    for n in tqdm(range(args.n_samples)):
        image = model.generate().to("cpu").detach().numpy() * 255
        image = np.squeeze(image).astype(np.uint8)
        image = Image.fromArray(image)
        image.threshold(thresholds=[254],values=[0,1])

        connZ = stat_fun(image, axis=2)
        cZ.append(connZ)
        cZ_mean = { k: cZ_mean.get(k, 0) + connZ.get(k, 0) for k in set(connZ) }
        
        connX = stat_fun(image, axis=1)
        cX.append(connX)
        cX_mean = { k: cX_mean.get(k, 0) + connX.get(k, 0) for k in set(connX) }
        
        connY = stat_fun(image, axis=0)
        cY.append(connY)
        cY_mean = { k: cY_mean.get(k, 0) + connY.get(k, 0) for k in set(connY) }

    cX_mean = {k: cX_mean.get(k,0)/args.n_samples for k in cX_mean.keys()}
    cY_mean = {k: cY_mean.get(k,0)/args.n_samples for k in cY_mean.keys()}
    cZ_mean = {k: cZ_mean.get(k,0)/args.n_samples for k in cZ_mean.keys()}
    categories = mpstool.connectivity.get_categories(image)

    vario_two = False # if variogram with two facies, plot only for one facies, since both variograms are identical
    if args.fun=="vario" and len(categories)==2:
        vario_two = True

    fig, axs = plt.subplots(len(categories), 3)

    for i, c in enumerate(categories):
        # X axis
        axs[i,0].fill_between(range(len(cY_ti_min[c])), cX_ti_min[c], cX_ti_max[c], color=GRAY)
        for smpl in cX:
            axs[i,0].plot(smpl[c], color='green', alpha=ALPHA)
        axs[i,0].plot(cX_mean[c], color='red')
        axs[i,0].plot(cX_ti_mean[c], color='black')
        title = "Orientation X" if vario_two else "Facies {}, Orientation X".format(i)
        axs[i,0].set_title(title)

        # Y axis
        axs[i,1].fill_between(range(len(cY_ti_min[c])), cY_ti_min[c], cY_ti_max[c], color=GRAY)
        for smpl in cY:
            axs[i,1].plot(smpl[c], color='green', alpha=ALPHA)
        axs[i,1].plot(cY_mean[c], color='red')
        axs[i,1].plot(cY_ti_mean[c], color='black')
        title = "Orientation X" if vario_two else "Facies {}, Orientation Y".format(i)
        axs[i,1].set_title(title)
        
        # Z axis
        axs[i,2].fill_between(range(len(cY_ti_min[c])), cZ_ti_min[c], cZ_ti_max[c], color=GRAY)
        for smpl in cZ:
            axs[i,2].plot(smpl[c], color='green', alpha=ALPHA)
        axs[i,2].plot(cZ_mean[c], color='red')
        axs[i,2].plot(cZ_ti_mean[c], color='black')
        title = "Orientation Z" if vario_two else "Facies {}, Orientation Z".format(i)
        axs[i,2].set_title(title)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    y_label = 'probability' if args.fun=="conn" else "$\gamma(t)$"
    for ax in axs.flat:
        ax.label_outer()
        ax.set(xlabel='distance (pixels)', ylabel=y_label)

    plt.show()