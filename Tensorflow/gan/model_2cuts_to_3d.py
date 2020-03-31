"""
WGAN generator and critic model for 50x50 images dataset
"""

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Activation
from keras.layers.convolutional import Conv2D as Conv2D, Conv2DTranspose as Conv2DT, Conv3D, Conv3DTranspose as Conv3DT
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

model_config = {
    "data_dimension" : (2,64,64),
    "generator_output_dimension" : (64,64,64),
    "noise_dimension" : (256,),
    "output_in_3D" : True,
    "cuts" : True,
    "scale_free" : False
}

def make_generator(noise_dim=model_config["noise_dimension"]):
    model = Sequential(name="generator")

    model.add( Reshape((4, 4, 4, 4), input_shape=noise_dim) )

    model.add( Conv3DT(512, (3, 3, 3), strides=2, padding='same', kernel_initializer='he_normal') )
    model.add( BatchNormalization(axis=1) )
    model.add( LeakyReLU(0.2) )

    model.add( Conv3DT(256, (5, 5, 5), strides=2, padding='same', kernel_initializer='he_normal') )
    model.add( BatchNormalization(axis=1) )
    model.add( LeakyReLU(0.2) )

    model.add( Conv3DT(128, (5, 5, 5), strides=2, padding='same', kernel_initializer='he_normal') )
    model.add( BatchNormalization(axis=1) )
    model.add( LeakyReLU(0.2) )

    model.add( Conv3DT(64, (5, 5, 5), strides=2, padding='same', kernel_initializer='he_normal') )
    model.add( BatchNormalization(axis=1) )
    model.add( LeakyReLU(0.2) )

    model.add( Conv3D(1, (5, 5, 5), padding='same', activation='tanh', kernel_initializer='he_normal') )

    model.add( Reshape((64,64,64)) )
    return model

def make_critic(input_dim=model_config["data_dimension"]):
    """ It is important in the WGAN-GP algorithm to NOT use batch normalization on the critic"""
    model = Sequential(name="critic")

    model.add( Reshape( (input_dim[0]*input_dim[1], input_dim[2], 1), input_shape=input_dim) )

    model.add( Conv2D(64, (5, 5), padding='same', strides=2))
    model.add( LeakyReLU(0.2) )

    model.add( Conv2D(128, (5, 5), kernel_initializer='he_normal', strides=2))
    model.add( LeakyReLU(0.2) )

    model.add( Conv2D(256, (5, 5), kernel_initializer='he_normal', padding='same', strides=2))
    model.add( LeakyReLU(0.2) )

    model.add( Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', strides=2))
    model.add( LeakyReLU(0.2) )

    model.add( Flatten() )

    model.add( Dense(1, kernel_initializer='he_normal'))
    return model
