# CS722 Course Project

This repository contains the code and data required to run my implementation of the Self-Supervised Semi-Supervised Learning (S4L) algortihm developed by Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer which can be found [here](https://arxiv.org/abs/1905.03670).

The code is all contained in one main Python script, so executing it is straightforward. It will save a log of the loss data in the data folder included in this repository, unless the filepath in the `save` function is changed. This script was written to work with the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), but with some slight modification it can be used as the basis for work on other image datasets. The code is heavily documented as to help with understanding of the project.

The number of layers can be changed following directions in the description accompanying the `config` method. The code also includes methods for producing datasets from the rotated images. When using the top level `train_model` method, be sure to carefully read the instructions and use a list of length two to specify which weight and bias layers you want the model to use as output. One should be for the supervised model and the other should be for one of the pretext tasks.
