# DiAGAN
3D simulations of earth textures using generative adversarial networks and 2D examples

## How to use

## Scripts

#### train.py

This script allow you to train a GAN.

Usage:
```
train.py [-h] --output_name OUTPUT_NAME --config CONFIG
         [--resume RESUME [RESUME ...]]

Various WGAN implementation for Keras.
```

Optional arguments :

```
  -h, --help            show this help message and exit
  --output_name OUTPUT_NAME, -out OUTPUT_NAME
                        The parameter file name to be output. Will be saved in
                        the 'models' folder
  --config CONFIG, -config CONFIG
                        The relative path to the config file you want to use
  --resume RESUME [RESUME ...], -r RESUME [RESUME ...]
                        --resume <epoch_to_resume> <path/to/generator>
                        <path/to/critic> Option to add if you want to resume
                        the training of a GAN.

```

Provided config files are located in the 'config' folder.
See the default.config file for an example.

Models will be saved in the __models__ subfolder, with extension .genmodel and .critmodel
Output images will be saved in the __output__ subfolder.
If they do not exist, those subfolders are created

#### generate.py

The `generate.py` file is meant to initialize a GAN from a model file (so, a GAN that has already been trained), and use its generator to create images.

Usage:
```
generate.py [-h] --model MODEL [--nb_image NB_IMAGE] [--vox] [--png]
                   [--tile]
```

Optional arguments:
```
  -h, --help            show this help message and exit
  --model MODEL, -mod MODEL
                        The model file to be loaded
  --nb_image NB_IMAGE, -n NB_IMAGE
                        The number of images to generate. Default value = 1
  --vox, -vox           Output in .vox file format
  --png, -png           Output in .png file format
  --tile, -t            Tile the output image in a square shape.
                        /!\ The images should be 2 dimensionnal /!\
```

Output images will be saved in the __output__ folder.

### Configuration file syntax

The script `train.py` has to be given a configuration file which should contain the following parameter s:
- *SOURCE* : The path to the training image(s)

- *ALGORITHM* : the training algorithm to use. Should be `GRADIENT_PENALTY` or `WEIGHT_CLIPPING`

- *MODEL* : The model to train. Available models are :
    -- 2d_50x50
    -- 2d_50x50_color
    -- 2d_256x256
    -- 2d_256x256_color
    -- 3d_50x50x50
    -- 3d_80x80x12
    -- 2cuts_to_3d
    -- 3cuts_to_3d

- *VERBOSE* : 0 or 1. If set to 1, enables verbose mode

- *SNAPSHOT* : 0 or 1. If set to 1, the program will save intermediate models at
               each epoch in the 'models' folder.

- *NUMBER_OF_EPOCHS* : Number of epochs in the training.
                       One epoch is one full pass over the training data set

- *EPOCH_SIZE* : Size of the set of examples when sampling
                 from a unigue large training image

- *BATCH_SIZE* : The batch size

- *N_CRITIC* : Number of training iteration of the critic network per iteration
               of the generator network.

- *MEMORY_ALLOC* : Tensorflow parameter. Set the fraction of the GPU memory used

- *LEARNING_RATE* : The learning rate of the GAN optimizer. Smaller learning rates lead to more stable but slower trainings.

- *MULTI_GPU* : 0 or 1. If set to 1, allows the program to run on multiple GPUs


### The input image

The input image is given to the programm through the configuration file.

For now, the program only reads .png and .gslib formats.
Two options are possible :

- Perform sampling of a large unique training image :
    in that case, the complete relative path to the image
    has to be given (e.g : path/to/training/image/TI.png)

- Use a precomputed set of training images :
    in that case, the images should be organized in the following way :
    ```
    /folder
        /png
            TI_1.png
            TI_2.png
            ...
        /gslib
            TI_1.gslib
            TI_2.gslib
            ...
     ```
Name of the training images inside the png or gslib subfolder do not matter.
It is possible to only have the /png or the /gslib folder.
If both folders are present, the program will only check for png/.

## Note on cut models
Cuts models take as input 2D images and output a 3D block of data which cuts
along the axis should ressemble the input examples.
Two models are available:
    - 2cuts_to_3d
    - 3cuts_to_3d
in 2cuts_to_3d, cuts are only made along the x and the y axis.
in 3cuts_to_3d, cuts are also made along the z axis.

To run a cut model, set the *MODEL* variable in the config file to a cut model,
then provide as the SOURCE variable a folder that contains two (or three) images named
x.png , y.png (and z.png) or x.gslib, y.gslib and (z.gslib)

Be careful of the orientation of your training images !

## Writing your own model architecture

This program allows you to write your own GAN architecture.
To do this, you need to create a file named `model_<yourArchitectureName>.py` in the gan package.

As the other model files, this file should provide three things :

- A parameter dictionnary called `model_config` containing the following keys :
    - 'data_dimension' : a tuple precising the shape of the input data
    - 'generator_output_dimension' : a tuple precising the shape of the generated data.
       It should be equal to 'data_dimension', unless 'cuts' is set to true
    - 'noise_dimension' : the input noise vector size. Should be a tuple of size 1 (e.g : (100,) )
    - 'output_in_3D' : boolean that indicates if the generator's output is in 3D or not
    - 'cuts' : a boolean that indicates if we perform 2D samples of the 3D
               output of the generator before piping into the critic

- A function called `make_generator`, returning the keras model defining the generator
- A function called `make_critic`, returning the keras model defining the critic

You can then train your model by setting the MODEL variable at `<yourArchitectureName>` in a configuration file.