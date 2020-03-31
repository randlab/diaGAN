"""
WGAN generator and critic model for a D 50x50x50 example set
"""

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv3D, Conv3DTranspose as Conv3DT
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

model_config = {
    "data_dimension" : (50,50,50),
    "generator_output_dimension" : (50,50,50),
    "noise_dimension" : (256,),
    "output_in_3D" : True,
    "scale_free" : False
}

def make_generator(noise_dim=model_config["noise_dimension"]):
    model = Sequential(name="generator")
    model.add( Dense(500, input_dim=noise_dim[0]))
    model.add( LeakyReLU(0.2))

    model.add( Reshape((5,5,5,4), input_shape=(500,)))

    model.add( Conv3DT(512, (5, 5, 5), strides=2, padding='same', kernel_initializer='he_normal'))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.2))

    model.add( Conv3DT(256, (5, 5, 5), strides=5, padding='same', kernel_initializer='he_normal'))
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
