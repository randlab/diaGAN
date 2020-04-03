import os
import numpy as np
import matplotlib.pyplot as plt
import mpstool
from mpstool.img import Image
from random import randint
import argparse
import keras
from gan.gan import GAN
from tqdm import tqdm

GRAY = '#a0a0a0'
ALPHA = 0.3

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-fun", "--fun", type=str, required=True)
    parser.add_argument('-model', "--model", type=str, required=True)
    parser.add_argument('-ti', "--ti", type=str, required=True)
    parser.add_argument('-output', "--output", type=str, default="output.csv")
    parser.add_argument("-n-samples", "--n-samples", type=int, default=30)
    parser.add_argument("-n-ti", "--n-ti", type=int, default=10)

    args = parser.parse_args()
    model = GAN(args.model)

    if ".vox" in args.ti:
        ti = Image.fromVox(args.ti).asArray()
    elif ".gslib" in args.ti:
        ti = Image.fromGslib(args.ti).asArray()

    if args.fun == "conn":
        stat_fun = mpstool.connectivity.get_function
    elif args.fun == "vario":
        stat_fun = mpstool.stats.variogram
    else:
        raise Exception("argument -fun not recognized. Should be 'conn' or 'vario' ")

     # Read connectivity for the TI : min and max of values
    cX_ti_min= {}
    cX_ti_max= {}

    cY_ti_min = {}
    cY_ti_max = {}

    cZ_ti_min = {}
    cZ_ti_max = {}

    sample_size = model.generate().shape

    for n in tqdm(range(args.n_ti)):
        ix = randint(0, ti.shape[0]-sample_size[0]-1)
        iy = randint(0, ti.shape[1]-sample_size[1]-1)
        iz = randint(0, ti.shape[2]-sample_size[2]-1)

        sample = ti[ix:ix+sample_size[0], iy:iy+sample_size[1], iz:iz+sample_size[2]]

        connX = stat_fun(sample, axis=1)
        cX_ti_min = { k: np.minimum(cX_ti_min.get(k, 1), connX.get(k, 1)) for k in set(connX) }
        cX_ti_max = { k: np.maximum(cX_ti_max.get(k, 0), connX.get(k, 0)) for k in set(connX) }

        connY = stat_fun(sample, axis=0)
        cY_ti_min = { k: np.minimum(cY_ti_min.get(k, 1), connY.get(k, 1)) for k in set(connY) }
        cY_ti_max = { k: np.maximum(cY_ti_max.get(k, 0), connY.get(k, 0)) for k in set(connY) }

        connZ = stat_fun(sample, axis=2)
        cZ_ti_min = { k: np.minimum(cZ_ti_min.get(k, 1), connZ.get(k, 1)) for k in set(connZ) }
        cZ_ti_max = { k: np.maximum(cZ_ti_max.get(k, 0), connZ.get(k, 0)) for k in set(connZ) }

    # Read connectivity for the realizations
    cX = []
    cY = []
    cZ = []
    cX_mean = {}
    cY_mean = {}
    cZ_mean = {}

    for n in tqdm(range(args.n_samples)):
        image = model.generate()
        image.threshold(thresholds=[1],values=[0,1])

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

    vario_two = False
    if args.fun=="vario" and len(categories)==2:
        categories= categories[:1]
        vario_two = True

    fig, axs = plt.subplots(len(categories), 3)

    for i, c in enumerate(categories):
        # X axis

        ax = axs[0] if vario_two else axs[i,0]
        ax.fill_between(range(len(cY_ti_min[c])), cX_ti_min[c], cX_ti_max[c], color=GRAY)
        for smpl in cX:
            ax.plot(smpl[c], color='green', alpha=ALPHA)
        ax.plot(cX_mean[c], color='red')
        title = "Orientation X" if vario_two else "Facies {}, Orientation X".format(i)
        ax.set_title(title)

        # Y axis
        ax = axs[1] if vario_two else axs[i,1]
        ax.fill_between(range(len(cY_ti_min[c])), cY_ti_min[c], cY_ti_max[c], color=GRAY)
        for smpl in cY:
            ax.plot(smpl[c], color='green', alpha=ALPHA)
        ax.plot(cY_mean[c], color='red')
        title = "Orientation X" if vario_two else "Facies {}, Orientation Y".format(i)
        ax.set_title(title)
        
        # Z axis
        ax = axs[2] if vario_two else axs[i,2]
        ax.fill_between(range(len(cY_ti_min[c])), cZ_ti_min[c], cZ_ti_max[c], color=GRAY)
        for smpl in cZ:
            ax.plot(smpl[c], color='green', alpha=ALPHA)
        ax.plot(cZ_mean[c], color='red')
        title = "Orientation Z" if vario_two else "Facies {}, Orientation Z".format(i)
        ax.set_title(title)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    y_label = 'probability' if args.fun=="conn" else "$\gamma(t)$"
    for ax in axs.flat:
        ax.label_outer()
        ax.set(xlabel='distance (pixels)', ylabel=y_label)

    plt.show()