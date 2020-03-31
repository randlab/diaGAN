#! /usr/bin/env python3
# coding: utf-8

import argparse
import os
from mpstool.img import *
from gan.gan import GAN
from keras.utils import Progbar

"""
Script generate.py
Used to initialize a GAN with existing model files, in order to generate output images.
"""

if __name__=="__main__":

    os.makedirs("models", exists_ok=True)
    os.makedirs("output", exists_ok=True)

    parser = argparse.ArgumentParser(description="Improved Wasserstein GAN \
                implementation for Keras.\n\
                Script generate.py\n\n\
                Used to initialize a GAN with existing model files, in order \
                to generate output images.")

    parser.add_argument('--model', '-mod', required=True, type=str,
                        help="The model file to be loaded")

    parser.add_argument('--nb_image', '-n', type=int, default=1,
                        help="The number of images to generate. Default value = 1")

    parser.add_argument('--size', '-s', nargs="+",
                        help="--size <x> <y> <z>\n\
                        Size of the images to be generated. \
                        Rounded to the nearest multiple of 10.\n\
                        This argument is ignored if the model does not \
                        allow variable sizes")

    parser.add_argument('--vox', '-vox', action="store_true",
                        help="Output in .vox file format")

    parser.add_argument('--png', '-png', action="store_true",
                        help="Output in .png file format")

    parser.add_argument('--tile', '-t', nargs="+",
                        help="--tile <nb_lines> <nb_columns>\n\
                        Tile the output images in the provided pattern")

    args = parser.parse_args()

    Gan = GAN(args.model)
    progbar = Progbar(args.nb_image)
    output_name = args.model.split("/")[-1].split(".")[0]
    if args.tile:
        assert len(args.tile)==2
        nx,ny = int(args.tile[0]), int(args.tile[1])
        output = []
        for i in range(args.nb_image):
            output.append(Gan.generate())
            progbar.add(1)
        tiled = Image.tile_images(output,(nx,ny))
        tiled.exportAsPng("output/{}_tile.png".format(output_name))
    else:
        for i in range(args.nb_image):
            img = Gan.generate()
            if args.png:
                img_name = "output/{}_{}.png".format(output_name, i)
                img.exportAsPng(img_name)
            elif args.vox:
                img_name = "output/{}_{}.vox".format(output_name, i)
                img.exportAsVox(img_name)
            else:
                img_name = "output/{}_{}.gslib".format(output_name, i)
                img.exportAsGslib(img_name)
            progbar.add(1)
