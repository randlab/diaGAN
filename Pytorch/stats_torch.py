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

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-model', "--model", type=str, required=True)
    parser.add_argument('-ti', "--ti", type=str, required=True)
    parser.add_argument('-output', "--output", type=str, default="output.csv")
    parser.add_argument("-n-samples", "--n-samples", type=int, default=30)
    parser.add_argument("-n-ti", "--n-ti", type=int, default=10)

    args = parser.parse_args()
    model = Generator(256).to("cpu")
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

   if ".vox" in args.ti:
        ti = Image.fromVox(args.ti).asArray()
    elif ".gslib" in args.ti:
        ti = Image.fromGslib(args.ti).asArray()

    if args.mode == "conn":
        stat_fun = mpstool.connectivity.get_function
    elif args.mode == "vario":
        stat_fun = mpstool.stats.variogram
    else:
        raise Exception("argument mode not recognized. Should be 'conn' or 'vario' ")

    # Read connectivity for the TI : min and max of values
    cX_ti_min= {}
    cX_ti_max= {}
    cX_ti_mean = {}

    cY_ti_min = {}
    cY_ti_max = {}
    cY_ti_mean = {}

    cZ_ti_min = {}
    cZ_ti_max = {}
    cZ_ti_mean = {}

    ti_size = ti.shape
    sample_size = model.generate().shape

    for n in tqdm(range(args.n_ti)):
        ix = randint(0, ti_size[0]-sample_size[0]-1)
        iy = randint(0, ti_size[1]-sample_size[1]-1)
        iz = randint(0, ti_size[2]-sample_size[2]-1)

        sample = ti[ix:ix+sample_size[0], iy:iy+sample_size[1], iz:iz+sample_size[2]]

        connX = stat_fun(sample, axis=1)
        cX_ti_min = { k: np.minimum(cX_ti_min.get(k, 1), connX.get(k, 1)) for k in set(connX) }
        cX_ti_max = { k: np.maximum(cX_ti_max.get(k, 0), connX.get(k, 0)) for k in set(connX) }
        cX_ti_mean = { k: cX_ti_mean.get(k, 0) + connX.get(k, 0) for k in set(connX) }

        connY = stat_fun(sample, axis=0)
        cY_ti_min = { k: np.minimum(cY_ti_min.get(k, 1), connY.get(k, 1)) for k in set(connY) }
        cY_ti_max = { k: np.maximum(cY_ti_max.get(k, 0), connY.get(k, 0)) for k in set(connY) }
        cY_ti_mean = { k: cY_ti_mean.get(k, 0) + connY.get(k, 0) for k in set(connY) }

        connZ = stat_fun(sample, axis=2)
        cZ_ti_min = { k: np.minimum(cZ_ti_min.get(k, 1), connZ.get(k, 1)) for k in set(connZ) }
        cZ_ti_max = { k: np.maximum(cZ_ti_max.get(k, 0), connZ.get(k, 0)) for k in set(connZ) }
        cZ_ti_mean = { k: cZ_ti_mean.get(k, 0) + connZ.get(k, 0) for k in set(connZ) }

    cX_ti_mean = {k: cX_ti_mean.get(k,0)/args.n_ti for k in cX_ti_mean.keys()}
    cY_ti_mean = {k: cY_ti_mean.get(k,0)/args.n_ti for k in cY_ti_mean.keys()}
    cZ_ti_mean = {k: cZ_ti_mean.get(k,0)/args.n_ti for k in cZ_ti_mean.keys()}

    # Read connectivity for the realizations
    cX = []
    cY = []
    cZ = []
    cX_mean = {}
    cY_mean = {}
    cZ_mean = {}

    for n in tqdm(range(args.n_samples)):
        image = model.generate()
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

    fig, axs = plt.subplots(len(categories), 3)

    for i, c in enumerate(categories):
        # X axis
        for smpl in cX:
            axs[i,0].plot(smpl[c], color='green')
        axs[i,0].plot(cX_mean[c], color='red')
        axs[i,0].plot(cX_ti_min[c], color='blue', linestyle="--")
        axs[i,0].plot(cX_ti_max[c], color='blue', linestyle="--")
        axs[i,0].plot(cX_ti_mean[c], color='blue')
        axs[i,0].set_title("Facies {}, Orientation X".format(i))

        # Y axis
        for smpl in cY:
            axs[i,1].plot(smpl[c], color='green')
        axs[i,1].plot(cY_mean[c], color='red')
        axs[i,1].plot(cY_ti_min[c], color='blue', linestyle="--")
        axs[i,1].plot(cY_ti_max[c], color='blue', linestyle="--")
        axs[i,1].plot(cY_ti_mean[c], color='blue')
        axs[i,1].set_title("Facies {}, Orientation Y".format(i))
        
        # Z axis
        for smpl in cZ:
            axs[i,2].plot(smpl[c], color='green')
        axs[i,2].plot(cZ_mean[c], color='red')
        axs[i,2].plot(cZ_ti_min[c], color='blue', linestyle="--")
        axs[i,2].plot(cZ_ti_max[c], color='blue', linestyle="--")
        axs[i,2].plot(cZ_ti_mean[c], color='blue')
        axs[i,2].set_title("Facies {}, Orientation Z".format(i))

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        ax.set(xlabel='distance (pixels)', ylabel='probability')

    plt.show()

