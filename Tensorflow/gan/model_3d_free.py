"""
WGAN generator and critic model for a D 50x50x50 example set
"""

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, GlobalMaxPooling3D
from keras.layers.convolutional import Conv3D, Conv3DTranspose as Conv3DT
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

model_config = {
    "data_dimension" : (50,50,50,1),
    "generator_output_dimension" : (50,50,50,1),
    "noise_dimension" : (5,5,5,1),
    "output_in_3D" : False,
    "scale_free" : True
}

def make_generator():
    model = Sequential(name="generator")

    model.add( Conv3DT(256, (5, 5, 5), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv3DT(512, (5, 5, 5), strides=5, padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv3DT(128, (5, 5, 5), padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv3D(1, (5, 5, 5), padding='same', activation='tanh'))

    model.add(Reshape(model_config["generator_output_dimension"]))
    return model

def make_critic(input_dim=model_config["data_dimension"]):
    """ It is important in the WGAN-GP algorithm to NOT use batch normalization on the critic"""
    model = Sequential(name="critic")
    model.add(Reshape((input_dim+(1,)), input_shape=input_dim))

    model.add(Conv3D(64, (5, 5, 5), padding='same'))
    model.add(LeakyReLU())

    model.add(Conv3D(128, (5, 5, 5), kernel_initializer='he_normal', strides=2))
    model.add(LeakyReLU())

    model.add(Conv3D(256, (5, 5, 5), kernel_initializer='he_normal', padding='same', strides=2))
    model.add(LeakyReLU())

    model.add(Conv3D(512, (5, 5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())

    model.add(Flatten())

    model.add(Dense(1, kernel_initializer='he_normal'))
    return model
