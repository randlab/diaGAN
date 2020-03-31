"""
WGAN generator and critic model for 50x50 images dataset
"""

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Activation
from keras.layers.pooling import GlobalMaxPooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D as Conv2D, Conv2DTranspose as Conv2DT
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

model_config = {
    "data_dimension" : (100,100,1),
    "generator_output_dimension" : (100,100,1),
    "noise_dimension" : (10,10,1),
    "output_in_3D" : False,
    "scale_free" : True
}

def make_generator(noise_dim=model_config["noise_dimension"]):
    model = Sequential(name="generator")

    # Noise upsampling
    model.add( Conv2DT(32, (5, 5), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv2DT(64, (5, 5), strides=5, padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    def add_block(n):
        model.add( Conv2D(n, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add( BatchNormalization())
        model.add( LeakyReLU(alpha=0.2))

    add_block(128)
    add_block(128)
    add_block(128)
    add_block(128)

    model.add( Conv2D(1, (3, 3), padding='same', activation='tanh', kernel_initializer='he_normal'))
    return model

def make_critic(noise_dim=model_config["noise_dimension"]):
    # It is important in the WGAN-GP algorithm to NOT use batch normalization on the critic
    model = Sequential(name="critic")

    def add_block(n):
        model.add(Conv2D(n, (3, 3), kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D())

    add_block(32)
    add_block(64)
    add_block(128)
    add_block(256)

    model.add(GlobalMaxPooling2D())
    model.add(Dense(1, kernel_initializer='he_normal'))
    return model
