"""
GAN generator and critic model for 250x250 images dataset
"""

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D as Conv2D, Conv2DTranspose as Conv2DT
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

model_config = {
    "data_dimension" : (256,256,3),
    "generator_output_dimension" : (256,256,3),
    "noise_dimension" : (256,),
    "output_in_3D" : False,
    "scale_free" : False
}

def make_generator(noise_dim=model_config["noise_dimension"]):
    # Creates a generator model that takes a 100-dimensional noise vector as a "seed"
    model = Sequential(name="generator")

    model.add(Dense(256, input_dim=noise_dim[0]))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((16,16,1), input_shape=noise_dim))

    model.add( Conv2DT(128, (7, 7), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(0.2))

    model.add( Conv2DT(256, (5, 5), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(0.2))

    model.add( Conv2DT(512, (5, 5), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(0.2))

    model.add( Conv2DT(256, (3, 3), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(0.2))

    model.add( Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(0.2))

    model.add( Conv2D(3, (5, 5), padding='same', activation='tanh', kernel_initializer='he_normal'))
    return model

def make_critic(input_dim=model_config["data_dimension"]):
    """ It is important in the WGAN-GP algorithm to NOT use batch normalization on the critic"""
    model = Sequential(name="critic")

    model.add(Conv2D(64, (4, 4), kernel_initializer='he_normal', padding='same', input_shape=input_dim, strides=[2, 2]))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, (4, 4), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, (4, 4), kernel_initializer='he_normal', strides=[2, 2]))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(512, (4, 4), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='he_normal'))
    return model
