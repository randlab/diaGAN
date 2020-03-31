import os
import numpy as np
from time import time
from math import *
from functools import partial

from keras.models import Model, load_model, save_model
from keras.layers import Input
from keras.optimizers import Adam
from keras.utils import Progbar
from keras import backend as K
import tensorflow as tf

from importlib import import_module
from mpstool.img import *
from .customLayers import *
from .utils import *

class GAN:
    """
    Implementation of the WGAN-GP algorithm
    Bibliography :
        - Initial Generative Adversarial Network exposal : https://arxiv.org/pdf/1406.2661.pdf
        - Wasserstein GAN : https://arxiv.org/abs/1701.07875
        - Improved training of WGAN (gradient penalty) : https://arxiv.org/abs/1704.00028

    Code adapted from : https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
    """

    ## --- Hyperparameters ---
    LAMBDA = 10 # gradient penalty weight

    # Adam optimizer parameters
    ADAM_BETA1 = 0.5
    ADAM_BETA2 = 0.9
    ADAM_EPSILON = 1e-8

    NOISE_TYPE = "NORMAL" # Choice between NORMAL and UNIF
    NORMAL_NOISE_MEAN = 0
    NORMAL_NOISE_STDV = 0.5
    UNIF_NOISE_MIN = 0
    UNIF_NOISE_MAX = 1

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
            self.starting_epoch=0
            self.batch_size = param["BATCH_SIZE"]
            self.verbose_mode = param["VERBOSE"]
            self.learning_rate = param["LEARNING_RATE"]
            self.n_critic = param["N_CRITIC"]
            self.multi_gpu = param["MULTI_GPU"]

            if self.verbose_mode:
                print("\n\n### Initializing a GAN with the following parameters :")
                print(" Algorithm : Gradient penalty")
                print(" Model : ", param["MODEL"])
                print(" Noise input dimension : " , self.noise_dim)
                if GAN.NOISE_TYPE=="NORMAL":
                    print(" Type of noise : NORMAL({},{})".format(GAN.NORMAL_NOISE_MEAN, GAN.NORMAL_NOISE_STDV))
                elif GAN.NOISE_TYPE=="UNIF":
                    print(" Type of noise : UNIFORM [{};{}]".format(GAN.UNIF_NOISE_MIN, GAN.UNIF_NOISE_MAX))
                print(" Data output dimension : ", self.data_dim)
                print(" Number of epochs : ", param["NUMBER_OF_EPOCHS"])
                print(" Batch size : ", param["BATCH_SIZE"])
                print(" n critic : ", self.n_critic)
                print(" Gradient penalty weight : ", GAN.LAMBDA)
                print(" Optimizer : Adam (alpha = {} ; beta1 = {} ; beta2 = {} ; epsilon = {})".format(self.learning_rate, GAN.ADAM_BETA1, GAN.ADAM_BETA2, GAN.ADAM_EPSILON))
                print(" Multi GPU support : {}".format(self.multi_gpu))
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
            self.optim = Adam(self.learning_rate,
                              beta_1=GAN.ADAM_BETA1,
                              beta_2=GAN.ADAM_BETA2,
                              epsilon=GAN.ADAM_EPSILON)

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
            if self.multi_gpu:
                self.gen_trainer = make_for_multi_gpu(self.gen_trainer)
            self.gen_trainer.compile(optimizer=self.optim,
                                     loss=wasserstein_loss)

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
            averaged_samples = AverageSampler(self.batch_size)([real_samples, generator_output_for_crit])
            critic_output_from_avg_samples = self.crit(averaged_samples)

            if self.multi_gpu:
                critic_output_from_generator = make_for_multi_gpu(critic_output_from_generator)
                critic_output_from_real_samples = make_for_multi_gpu(critic_output_from_real_samples)
                # For some reason, we are unable to compute the gradient penalty loss on a multi GPU model

            gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples)

            self.crit_trainer = Model(inputs=[real_samples, generator_input_for_crit],
                                      outputs=[critic_output_from_real_samples,
                                               critic_output_from_generator,
                                               critic_output_from_avg_samples],
                                      name="global")
            self.crit_trainer.compile(optimizer=self.optim,
                                      loss=[wasserstein_loss,
                                            wasserstein_loss,
                                            gp_loss],
                                      loss_weights=[1, 1, GAN.LAMBDA])
            if self.verbose_mode:
                visualize_model(self.crit_trainer)
            print("Initialization succesful")

        else :
            assert(isinstance(arg[0],str))
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

    ## Save and load functions
    def save_model(self, filename, save_critic=True):
        """
        Save both the generator and the critic model parameters onto the disk.
        """
        self.gen.save(os.path.join("models",filename+".genmodel"))
        if save_critic:
            self.crit.save(os.path.join("models",filename+".critmodel"))
        print("Model {} has been saved in the 'models' folder".format(filename))

    def generate(self, seed=None):
        """Feeds random seeds into the generator """
        img = self.gen.predict(self.get_noise(1)) if seed is None else self.gen.predict(seed)
        img = np.squeeze(img)
        img = Image.fromArray(img)
        img.unnormalize()
        return img

    def take_snapshot(self, i_epoch, snapshot=True, tiled=True):
        """
        Save the current state of the weights of the model, and generate some samples along the way
        """
        if snapshot:
            self.save_model("epoch_{}".format(i_epoch))
        samples = []
        if self.model_config["output_in_3D"]:
            img = self.generate()
            img.exportAsGslib("output/epoch_{}.gslib".format(i_epoch), verbose=self.verbose_mode)
            img.exportAsVox("output/epoch_{}.vox".format(i_epoch), verbose=self.verbose_mode)
        else :
            if tiled:
                for i in range(10):
                    samples.append(self.generate())
                tiled = Image.tile_images(samples, mode="h")
                tiled.exportAsPng("output/epoch_{}.png".format(i_epoch), verbose=self.verbose_mode)
            else:
                self.generate().exportAsPng("output/epoch_{}.png".format(i_epoch), verbose=self.verbose_mode)

    ## Training functions
    def get_noise(self, size):
        """
        Samples an input vector from the GAN.
        Depending on the GAN.NOISE_TYPE global parameter, will generate along a uniform or a normal distribution
        """
        if GAN.NOISE_TYPE=="UNIF":
            return np.random.rand(size, self.noise_dim[0]).astype(np.float32) # UNIFORM
        elif GAN.NOISE_TYPE=="NORMAL":
            return np.random.normal(GAN.NORMAL_NOISE_MEAN, GAN.NORMAL_NOISE_STDV, (size, self.noise_dim[0])).astype(np.float32) #NORMAL

    def get_label(self,size):
        """
        Samples labels from a normal law of mean 1 and standard deviation 1e-3
        """
        return np.random.normal(1, 1e-3, size).astype(np.float32)

    def get_batch(self, source, i, size):
        return source[i*size:(i+1)*size]

    def train(self, examples, snapshot=False):
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
        dummy = np.zeros((MINIBATCH_SIZE, 1), dtype=np.float32) #given to the gradient penalty loss, but not used
        start_time = time()

        for i_epoch in range(self.starting_epoch, self.nb_epochs):
            epoch_time = time()
            np.random.shuffle(examples)
            progbar = Progbar(EPOCH_SIZE)
            print("Epoch {}/{} :".format(i_epoch+ 1, self.nb_epochs))
            for i_batch in range(NB_BATCH):
                batch = self.get_batch(examples, i_batch, BATCH_SIZE)
                critic_loss = []
                generator_loss = []
                for i_minibatch in range(self.n_critic):
                    labels_for_real = self.get_label((MINIBATCH_SIZE,1))
                    labels_for_generated = -self.get_label((MINIBATCH_SIZE,1))
                    minibatch = self.get_batch(batch, i_minibatch, MINIBATCH_SIZE)
                    loss =self.crit_trainer.train_on_batch(
                             [minibatch, self.get_noise(MINIBATCH_SIZE)],
                             [labels_for_real, labels_for_generated, dummy])
                    critic_loss.append(loss)
                loss = self.gen_trainer.train_on_batch(
                            self.get_noise(MINIBATCH_SIZE),
                            self.get_label((MINIBATCH_SIZE,1)))
                generator_loss.append(loss)
                progbar.add(BATCH_SIZE, values=[("Loss_critic", np.mean(critic_loss)-np.mean(generator_loss)),
                                                ("Loss_generator", -np.mean(generator_loss))])
            print("Time: %.2fs, Total time : %.2fs\n" % (time()-epoch_time, time()-start_time))
            self.take_snapshot(i_epoch+1, snapshot=snapshot, tiled=True)
        print("Training complete")
