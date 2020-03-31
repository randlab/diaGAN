import os
import numpy as np
from time import time
from math import *
from functools import partial

from keras.models import Model, load_model, save_model
from keras.layers import Input
from keras.optimizers import RMSprop
from keras.utils import Progbar
from keras import backend as K

from importlib import import_module
from mpstool.img import *
from .customLayers import *
from .utils import *
from .gan import GAN

def clip_weights(critic, min_value, max_value):
    """ 
    Clipping the weight is the heart of this algorithm
    -> back to a case where our weights lay in a compact
    """
    for layer in critic.layers:
        weights = layer.get_weights()
        weights = [np.clip(w, min_value, max_value) for w in weights]
        layer.set_weights(weights)

class WGAN(GAN):
    """
    Implementation of the WGAN algorithm
    Bibliography :
        - Initial Generative Adversarial Network exposal : https://arxiv.org/pdf/1406.2661.pdf
        - Wasserstein GAN : https://arxiv.org/abs/1701.07875
    """

    ## --- Hyperparameters ---

    C = 0.01 # gradient penalty weight

    def __init__(self, *arg):
        """
        Initialization method.
        There are two ways a GAN can be initiated :

        1) Given a dictionnary of parameters as argument. This dictionnary should contain the following keys :


         (see train.py for an example of such a dictionnary)


        2) Given one or two keras model file names.
           When two models are given, ***the generator model should be the first one***, followed by the critic
           When only one model is given, it is assumed that is it a generator
        """

        if isinstance(arg[0],dict):
            param = arg[0] # arg is a parameter dict for training the GAN

            # Initialize parameters
            model_to_import = ".model_"+param["MODEL"]
            model = import_module(model_to_import,package="gan")

            self.model_config = model.model_config
            self.data_dim = self.model_config["data_dimension"]
            self.noise_dim = self.model_config["noise_dimension"]
            self.nb_epochs = param["NUMBER_OF_EPOCHS"]
            self.starting_epoch = 0
            self.n_critic = param["N_CRITIC"]
            self.learning_rate = param["LEARNING_RATE"]
            self.batch_size = param["BATCH_SIZE"]
            self.verbose_mode = param["VERBOSE"]

            if self.verbose_mode:
                print("\n\n### Initializing a GAN with the following parameters :")
                print(" Algorithm : weight clipping (Wasserstein GAN)")
                print(" Model : ", param["MODEL"])
                print(" Noise input dimension : " , self.noise_dim)
                if WGAN.NOISE_TYPE=="NORMAL":
                    print(" Type of noise : NORMAL({},{})".format(GAN.NORMAL_NOISE_MEAN, GAN.NORMAL_NOISE_STDV))
                elif WGAN.NOISE_TYPE=="UNIF":
                    print(" Type of noise : UNIFORM [{};{}]".format(GAN.UNIF_NOISE_MIN, GAN.UNIF_NOISE_MAX))
                print(" Data output dimension : ", self.data_dim)
                print(" Number of epochs : ", param["NUMBER_OF_EPOCHS"])
                print(" Batch size : ", param["BATCH_SIZE"])
                print(" n critic : ", self.n_critic)
                print(" Clipping parameter : ", WGAN.C)
                print(" Optimizer : RMSProp (learning rate = {})".format(self.learning_rate))
                print(" Multi GPU support : {}".format(param["MULTI_GPU"]))
                print("\n\n")

            # Initialize models
            if len(arg)==4:
                self.starting_epoch = arg[3]
                generator_name, critic_name = arg[1], arg[2]
                print("Loading generator model from file {}".format(generator_name))
                self.gen = load_model(generator_name)
                print("Loading critic model from file {}".format(critic_name))
                self.crit = load_model(critic_name)
            else:
                self.gen = model.make_generator(self.noise_dim)
                self.crit = model.make_critic(self.data_dim)
            if self.verbose_mode:
                visualize_model(self.gen)
                visualize_model(self.crit)

            # Initialize the Adam optimizer
            self.optim = RMSprop(lr=self.learning_rate)

            # Compile the generator with fixed critic
            for layer in self.crit.layers:
                layer.trainable = False
            self.crit.trainable = False

            generator_input = Input(shape=self.noise_dim)
            generator_output = self.gen(generator_input)

            if "cuts" in self.model_config and self.model_config["cuts"]:
                n_cuts = self.data_dim[0]
                generator_output = CutSampler(n_cuts)(generator_output)

            critic_output = self.crit(generator_output)
            self.gen_trainer = Model(inputs=[generator_input], outputs=[critic_output])
            if param["MULTI_GPU"]:
                self.gen_trainer = make_for_multi_gpu(self.gen_trainer)
            self.gen_trainer.compile(optimizer=self.optim, loss=wasserstein_loss)

            # compile critic with fixed generator
            for layer in self.crit.layers:
                layer.trainable = True
            self.crit.trainable = True

            for layer in self.gen.layers:
                layer.trainable = False
            self.gen.trainable = False

            real_samples = Input(shape=self.data_dim)
            generator_input_for_crit = Input(shape=self.noise_dim)
            generator_output_for_crit = self.gen(generator_input_for_crit)

            critic_output_from_generator = self.crit(generator_output_for_crit)
            critic_output_from_real_samples = self.crit(real_samples)

            self.crit_trainer = Model(inputs=[real_samples, generator_input_for_crit],
                                      outputs=[critic_output_from_real_samples,
                                               critic_output_from_generator],
                                      name="global")
            self.crit_trainer.output_names = ['output_real', 'output_gen']
            if param["MULTI_GPU"]:
                self.crit_trainer = make_for_multi_gpu(self.crit_trainer)
            self.crit_trainer.compile(optimizer=self.optim,
                                      loss=[wasserstein_loss,
                                            wasserstein_loss])
            if self.verbose_mode:
                visualize_model(self.crit_trainer)
            print("Initialization succesful")
        else :
            # load GAN from a file
            generator_name = arg[0]
            print("Loading generator model from file {}".format(generator_name))
            self.gen = load_model(generator_name)
            self.noise_dim = self.gen.input_shape[1:]
            self.data_dim = self.gen.output_shape

            if len(arg)>1:
                critic_name = arg[1]
                print("Loading critic model from file {}".format(critic_name))
                self.crit = load_model(critic_name)
            print("Loading succesful")


    def train(self, examples, snapshot=True):
        """
        Trains the model on the 'examples' dataset.
        """
        print("\n\nStarting training of GAN")
        if snapshot:
            print("Models will be saved along the training in the 'models' folder")
        else:
            print("/!\ Snapshot option is disabled : no intermediate model will be saved.\nTo activate snapshot, use the -sn (or --snapshot) option")


        BATCH_SIZE = self.batch_size*self.n_critic
        MINIBATCH_SIZE = self.batch_size
        NB_BATCH = examples.shape[0]//BATCH_SIZE
        EPOCH_SIZE = NB_BATCH*BATCH_SIZE # we dump the eventual non-complete batch at the end
        start_time = time()

        for i_epoch in range(self.starting_epoch, self.nb_epochs):
            epoch_time = time()
            np.random.shuffle(examples)
            progbar = Progbar(EPOCH_SIZE)
            for i_batch in range(NB_BATCH):
                batch = self.get_batch(examples, i_batch, BATCH_SIZE)
                critic_loss = []
                generator_loss = []
                for i_minibatch in range(self.n_critic):
                    labels_for_real = self.get_label((MINIBATCH_SIZE,1))
                    labels_for_generated = -self.get_label((MINIBATCH_SIZE,1))
                    minibatch = self.get_batch(batch, i_minibatch, MINIBATCH_SIZE)
                    loss = self.crit_trainer.train_on_batch(
                                [minibatch, self.get_noise(MINIBATCH_SIZE)],
                                [labels_for_real, labels_for_generated])
                    critic_loss.append(loss)
                clip_weights(self.crit,-WGAN.C,WGAN.C)
                loss = self.gen_trainer.train_on_batch(
                            self.get_noise(MINIBATCH_SIZE),
                            self.get_label((MINIBATCH_SIZE,1)))
                generator_loss.append(loss)
                progbar.add(BATCH_SIZE, values=[("Loss_critic", np.mean(critic_loss)-np.mean(generator_loss)),
                                                ("Loss_generator", -np.mean(generator_loss))])
            print("\nEpoch %s/%s, Time: %.2fs, Total time : %.2fs\n" % (i_epoch+ 1, self.nb_epochs, time()-epoch_time, time()-start_time))
            self.take_snapshot(i_epoch+1, snapshot=snapshot, tiled=True)
        print("Training complete")
