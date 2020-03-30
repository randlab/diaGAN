# DiAGAN
3D simulations of earth textures using generative adversarial networks and 2D examples

Code accompagning this article : *TODO*

For pretrained models, training images and sample of outputs, see https://github.com/UniNE-CHYN/DiAGAN_Examples

### References
- [Generative Adversarial Nets, Goodfellow et al.](https://arxiv.org/pdf/1406.2661.pdf)
- [Wasserstein GAN, Arjovsky et al.](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs, Guljarani et al.](https://arxiv.org/abs/1704.00028)

### Dependencies
python (>3.5) with the following librairies :
  - Pillow
  - Numpy
  - Tensorflow
  - Keras
  - mpstool : https://github.com/UniNE-CHYN/mps_toolbox
  - py-vox-io (optionnal) : https://github.com/gromgull/py-vox-io

## Principle

We added a random cut sampler between the output of the generator and the input of the discriminator. This allow us to generate 3D examples, but feed the discriminator with generated elements that have the same shape as example elements.

![GAN architecture for 2D to 3D synthesis](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/gan3D.png)

The algorithm can also work without this additionnal element in the architecture, which gives a classical WGAN implementing the gradient penalty loss.

## How to use
See the `HOW_TO_USE.md` file

## Image Gallery
### 2D synthesis

| Training examples | Output images (tiled) |
|-------------------|---------------|
|<img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/strebelle2500.png" width=300> | <img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/strebelle256_tile_small.png" width=300>|
|<img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/beton1500.png" width=300> | <img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/beton_tile_small.png" width=300> |
|<img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/karst_large.png" width=300> | <img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/karst_tile_small.png" width=300> |
|<img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/fracture.png" width=300> | <img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/fractures_tile_small.png" width=300>|
|<img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/braided.png" width=300> | <img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/braided_tile_small.png" width=300> |
|<img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/stone_colored.png" width=300>| <img src="https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/colored_tile_small.png" width=300> |

### 3D synthesis out of 3D examples

| Training examples | Output images | Output images |
|-------------------|---------------|---------------|
|![Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/channels.png)  |   ![Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/channel1_1.png)| ![Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/channel2_1.png) |

### 3D synthesis out of 2D examples

examples in the list below were fed to the GAN as cuts along three axis.

| Training examples | Output images | Output images |
|-------------------|---------------|---------------|
| ![Strebelle Channels TI](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/strebelle.png) ![Strebelle Channels TI](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/strebelle_z.png) | ![Strebelle Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/strebelle1.png) ![Strebelle Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/strebelle3.png) | ![Strebelle Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/strebelle2.png) ![Strebelle Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/strebelle4.png) |
| ![houthuys TI x](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/houthuys_x.png) ![houthuys TI y](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/houthuys_y.png) | ![houthuys](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/houthuys1_1.png) ![houthuys](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/houthuys1_2.png) | ![houthuys](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/houthuys2_1.png) ![houthuys](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/houthuys2_2.png)|
|![F42A](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/f42a.png) | ![F42A](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/cube1_1.png)| ![F42A](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/cube2_1.png)|
| ![Balls](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/balls.png) | ![Balls](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/balls_cuts1_1.png) | ![Balls](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/balls_cuts2_1.png) |
