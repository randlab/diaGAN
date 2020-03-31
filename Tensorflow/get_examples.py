import os
from os.path import join as pj
import numpy as np
from mpstool.img import *
from copy import deepcopy

SAMPLE_RATE=10000

def sample_from_png(filename, param):
    assert(".png" in filename)
    nb_channels = param["DATA_DIM"][2]
    channel_mode = "RGB" if nb_channels in [3,4] else "Gray"
    source = Image.fromPng(filename, channel_mode=channel_mode)
    if param["VERBOSE"]:
        print("Source image shape : ", source.shape)

    if nb_channels==1:
        output = [source.get_sample(param["DATA_DIM"], normalize=True) for i in range(param["EPOCH_SIZE"])]
    else:
        output = [source.get_sample(param["DATA_DIM"], normalize=True, var_name=["R","G","B"]) for i in range(param["EPOCH_SIZE"])]

    if param["VERBOSE"]:
        for i in range(int(len(output)//SAMPLE_RATE)):
            to_output = deepcopy(output[i])
            to_output.exportAsPng("output/test_example{}.png".format(i))

    output = np.array([img.asArray() for img in output])
    return output


def sample_from_gslib(filename, param):
    assert(".gslib" in filename)
    source = Image.fromGslib(filename)
    if param["VERBOSE"]:
        print("source image shape : ", source.shape)
    output = [source.get_sample(param["DATA_DIM"], normalize=True) for i in range(param["EPOCH_SIZE"])]
    if param["VERBOSE"]:
        for i in range(int(len(output)//SAMPLE_RATE)):
            to_output = deepcopy(output[i])
            to_output.exportAsVox("output/test_example{}.vox".format(i))
    output = np.array([img.asArray() for img in output])
    return output


def get_train_examples(param):
    examples = []
    path = pj(os.getcwd(),param["SOURCE"])

    if '.png' in path:
        return sample_from_png(path, param)
    elif '.gslib' in path:
        return sample_from_gslib(path, param)

    print("Using examples from the '{}' folder".format(param["SOURCE"]))
    path_png = pj(path,"png")
    path_gslib = pj(path,"gslib")

    # Loading from png subfolder
    if os.path.exists(path_png):
        for f in os.listdir(path_png):
            print("Reading file {} ...".format(f))
            image = Image.fromPng(pj(path_png,f), normalize=True)# data are in range [-1,1]
            examples.append(image.asArray())
        return np.array(examples)

    # Loading from gslib subfolder
    elif os.path.exists(path_gslib):
            path_gslib = pj(path,"gslib")
            for f in os.listdir(path_gslib):
                print("Reading file {} ...".format(f))
                image = Image.fromGslib(pj(path_gslib,f), normalize=True)
                # data are in range [-1,1]
                examples.append(image.asArray())
            return np.array(examples)

def sample_cuts_from_gslib(filename, param, nb_cuts):
    # Assumes the gslib represents some 3D data
    assert(".gslib" in filename)
    source = Image.fromGslib(filename)
    if param["VERBOSE"]:
        print("source image shape : ", source.shape)
    output = []
    size =  param["DATA_DIM"][1]
    xs,ys,zs = source.shape
    source = source.asArray()
    for i in range(param["EPOCH_SIZE"]):
        sample = []
        x = np.random.randint(xs-size)
        y = np.random.randint(ys-size)
        z = np.random.randint(zs-size)
        sample.append(source[x, y:y+size, z:z+size])
        sample.append(source[x:x+size, y, z:z+size])
        if nb_cuts==3:
             sample.append(source[x:x+size, y:y+size, z])
        sample = Image.fromArray(np.array(sample))
        sample.normalize()
        output.append(sample.asArray())
    output = np.array(output)
    if param["VERBOSE"]:
        print("Example tensor shape :", output.shape)
        for i in range(int(len(output)//SAMPLE_RATE)):
            img = deepcopy(output[i*SAMPLE_RATE,:,:,:])
            img = np.concatenate(img, axis=0)
            img = img.reshape(img.shape+(1,))
            Image.fromArray(img).exportAsPng("output/test_example{}.png".format(i))
    return output


def get_train_examples_cuts(param, nb_cuts):
    path = param["SOURCE"]
    assert nb_cuts==2 or nb_cuts==3
    if '.gslib' in path:
        return sample_cuts_from_gslib(path, param, nb_cuts)

    # Sampling from png training image
    source_x = Image.fromPng(pj(path,"x.png"))
    source_y = Image.fromPng(pj(path,"y.png"))
    source_z = Image.fromPng(pj(path,"z.png"))
    output=[]
    extract_dim = param["DATA_DIM"][1:]+(1,)
    for i in range(param["EPOCH_SIZE"]):
        sample_x = source_x.get_sample(extract_dim, normalize=True).asArray()
        sample_x = sample_x.reshape(sample_x.shape[:-1])
        sample_y = source_y.get_sample(extract_dim, normalize=True).asArray()
        sample_y = sample_y.reshape(sample_y.shape[:-1])
        sample = [sample_x, sample_y]
        if nb_cuts==3:
            sample_z = source_z.get_sample(extract_dim, normalize=True).asArray()
            sample_z = sample_z.reshape(sample_z.shape[:-1])
            sample.append(sample_z)
        sample = np.array(sample)
        output.append(sample)
    output = np.array(output)
    if param["VERBOSE"]:
        print("Example tensor shape :", output.shape)
        for i in range(int(len(output)//SAMPLE_RATE)):
            img = deepcopy(output[i*SAMPLE_RATE,:,:])
            img = np.concatenate(img, axis=0)
            img = img.reshape(img.shape+(1,))
            Image.fromArray(img).exportAsPng("output/test_example{}.png".format(i))
    return output
