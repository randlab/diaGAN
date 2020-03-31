"""
Temporary utilitary file to generate models
"""

from model_3d_80x80x12 import *

def visualize_model(model):
    print("----- MODEL SUMMARY : {} -----".format(model.name))
    model.summary()
    from keras.utils import plot_model
    plot_model(model,
               to_file='{}_summary.png'.format(model.name),
               show_shapes=True,
               show_layer_names=True)
    print("\n\n")

if __name__=="__main__":
    model_gen = make_generator()
    visualize_model(model_gen)
    model_crit = make_critic()
    visualize_model(model_crit)
