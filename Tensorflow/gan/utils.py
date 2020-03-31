from keras.utils import multi_gpu_model
from keras import backend as K
import numpy as np
import os

def eye_loss(y_real, y_pred):
    """ Identity loss """
    return K.sum(y_pred)

def wasserstein_loss(y_real, y_pred):
    """
    The original Wasserstein GAN loss function
    """
    return K.mean(y_real * y_pred)

def gradient_penalty_loss(y_real, y_pred, averaged_samples):
    """
    Computes the gradient penalty term in the WGAN-GP loss function
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(gradient_l2_norm-1)
    return K.mean(gradient_penalty)

def visualize_model(model):
    """
    Given a keras model, outputs its summary in the console.
    Also generates a .png summary file with graphviz
    """
    print("\n\n----------- MODEL SUMMARY : {} -----------\n".format(model.name))
    model.summary()
    try:
        from keras.utils import plot_model
        filename = os.path.join('models', '{}_summary.png'.format(model.name))
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
        print("Model summary have been saved in {}\n\n".format(filename))
    except Exception as e:
        print("Something went wrong when trying to plot the model :")
        print(e)
        print("Ignoring this part and continuing execution")

def make_for_multi_gpu(model):
    try:
        return multi_gpu_model(model)
    except Exception as e:
        print("Something went wrong when trying to make a multi_gpu_model out of the generator : ")
        print(e)
        if "gpus=1" or "gpus=0" in str(e):
            print("To get rid of this error message, disable the MULTI_GPU option in the config file")
        print("Multi GPU training has been disabled for this session.")
        return model
