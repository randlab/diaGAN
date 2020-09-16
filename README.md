# DiAGAN
Code accompagning the article `3D Geological Image Synthesis From 2D Examples Using Generative Adversarial Networks`, Guillaume Coiffier, Philippe Renard and Sylvain Lefebvre

Generative Adversarial Networks (GAN) are becoming an alternative to Multiple-point Statistics (MPS) techniques to generate stochastic fields from training images. But a difficulty for all the training image based techniques (including GAN and MPS) is to generate 3D fields when only 2D training data sets are available. In this paper, we introduce a novel approach called Dimension Augmenter GAN (DiAGAN) enabling GANs to generate 3D fields from 2D examples. The method is simple to implement and is based on the introduction of a random cut sampling step between the generator and the discriminator of a standard GAN.

![GAN architecture for 2D to 3D synthesis](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/gan3D.png)

For pretrained models, training images and sample of outputs, see https://github.com/randlab/DiAGAN_Examples

### Dependencies
python (>3.5) with the following librairies :
  - Pillow
  - Numpy
  - mpstool : https://github.com/UniNE-CHYN/mps_toolbox
  - py-vox-io (optionnal) : https://github.com/gromgull/py-vox-io
  - Tensorflow (for the TF version)
  - Keras (for the TF version)
  - Pytorch (for the pytorch version)

## How to use
See the `HOW_TO_USE.md` file

## Image Gallery

| Training examples | Output images | Output images |
|-------------------|---------------|---------------|
| ![Strebelle Channels TI](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/strebelle.png) ![Strebelle Channels TI](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/strebelle_z.png) | ![Strebelle Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/strebelle1.png) ![Strebelle Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/strebelle3.png) | ![Strebelle Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/strebelle2.png) ![Strebelle Channels](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/strebelle4.png) |
| ![houthuys TI x](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/houthuys_x.png) ![houthuys TI y](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/houthuys_y.png) | ![houthuys](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/houthuys1_1.png) ![houthuys](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/houthuys1_2.png) | ![houthuys](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/houthuys2_1.png) ![houthuys](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/houthuys2_2.png)|
|![F42A](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/f42a.png) | ![F42A](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/cube1_1.png)| ![F42A](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/cube2_1.png)|
| ![Balls](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/ti/balls.png) | ![Balls](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/balls_cuts1_1.png) | ![Balls](https://github.com/UniNE-CHYN/DiAGAN_Examples/blob/master/gallery/balls_cuts2_1.png) |
