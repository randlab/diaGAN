"""
WGAN generator and critic model for 50x50 images dataset
"""

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Activation
from keras.layers.convolutional import Conv2D as Conv2D, Conv2DTranspose as Conv2DT
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

model_config = {
    "data_dimension" : (50,50,3),
    "generator_output_dimension" : (50,50,3),
    "noise_dimension" : (100,),
    "output_in_3D" : False,
    "scale_free" : False
}

def make_generator(noise_dim=model_config["noise_dimension"]):
    model = Sequential(name="generator")

    model.add(Dense(200, input_dim=noise_dim[0]))
    model.add(LeakyReLU(alpha=0.2))

    model.add( Reshape((5, 5, 8), input_shape=noise_dim))

    model.add( Conv2DT(512, (5, 5), strides=2, padding='same'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv2DT(256, (5, 5), strides=5, padding='same'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv2D(128, (5, 5), padding='same'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv2DT(64, (5, 5), padding='same'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv2D(3, (5, 5), padding='same', activation='tanh'))
    return model


def make_critic(input_dim=model_config["data_dimension"]):
    # It is important in the WGAN-GP algorithm to NOT use batch normalization on the critic
    model = Sequential(name="critic")
    model.add(Conv2D(256, (3, 3), padding='same', input_shape=input_dim))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(64, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())

    model.add(Dense(1, kernel_initializer='he_normal'))
    return model
