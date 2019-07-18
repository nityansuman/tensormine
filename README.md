<p align="center"><img src="data/logo.png?raw=true" alt="LOGO"/></p>

<img alt="PyPI" src="https://img.shields.io/pypi/v/tensorhub.svg?color=blue&style=flat"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/tensorhub.svg?style=flat">  <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/nityansuman/tensorhub.svg?color=blue&style=flat"> <img alt="PyPI - License" src="https://img.shields.io/pypi/l/tensorhub.svg?style=flat"> [![Codacy Badge](https://api.codacy.com/project/badge/Grade/d1e35c252db741b28144f5b7b9ffd7d2)](https://www.codacy.com/app/nityansuman/tensorhub?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nityansuman/tensorhub&amp;utm_campaign=Badge_Grade) [![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors)



## You have just found TensorHub.

The open source library to help you develop and train ML models, easy and fast as never before with `TensorFlow 2.0`.
`TensorHub` is a global collection of `building blocks` and `ready to serve models`.

It is a wrapper library of deep learning models and neural lego blocks designed to make deep learning more accessible and accelerate ML research.

Use `TensorHub` if you need a deep learning library that:

+ **Reproducibility** - Reproduce the results of existing pre-training models (such as Google BERT, XLNet)

+ **Model modularity** - TensorHub divided into multiple components: ready-to-serve models, layers, neural-blocks etc. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules with as few restrictions as possible.

+ **Prototyping** - Code less build more. Apply `TensorHub` to create fast prototypes with the help of modulear blocks, custom layers, custom activation support.

+ **Platform Independent** - Supports both `Keras` and `TensorFlow 2.0`. Run your model on CPU, single GPU or using a distributed training strategy.

------------------


## Getting started: 30 seconds to TensorHub

<p align="center"><img src="data/readme_code_example_1.png?raw=true" alt="code"/></p>

Building a question answering system, an image classification model, a Neural Turing Machine, or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

For a more in-depth tutorial about Keras, you can check out:

+ [Getting started with Text Classification](https://github.com/nityansuman/tensorhub/tree/master/examples/training-a-text-classifier-using-tensorhub-models.ipynb)
+ [Getting started with Custom Layers](https://github.com/nityansuman/tensorhub/tree/master/examples/creating-custom-models.ipynb)

In the [examples folder](https://github.com/nityansuman/tensorhub/tree/master/examples) of this repository, you will find much more advanced examples coming your way very soon.

------------------


## What's coming in V1.0
+ Cooked Models
    + Image Classification (Supports Transfer Learning with ImageNet Weights)
        + Xception
        + VGG16
        + VGG19
        + ResNet50
        + InceptionV3
        + InceptionResNetV2
        + MobileNet
        + DenseNet
        + NASNet
        + SqueezeNet (Without Transfer Learning) *

    + Text Classification
        + RNN Model
        + LSTM Model
        + GRU Model
        + Text-CNN

    + Neural Machine Translation *
        + Encoder-Decoder Sequence Translation Model
        + Translation with Attention

    + Text Generation *
        + RNN, LSTM, GRU Based Model
        
    + Named Entity Recogniton *
        + RNN, LSTM, GRU Based Model

+ Custom Modules/Layers
    + Standard Layers
        + Linear
        + Inception V1 Layer
        + Inception V1 with Reduction Layer
        + Inception V2 Layer *
        + Inception V3 Layer *
    + Attention layers
        + Bahdanau Attention
        + Luong Attention
        + Self-Attention *

+ Utilities
    + Text
        + Custom Word and Character Tokenizer
        + Load Pre-trained Embeddings
        + Create Vocabulary Matrix
    + Image *
        + Image Augmentation
    + Activations
        + RELU
        + SELU
        + GELU
        + ELU
        + Tanh
        + Sigmoid
        + Hard Sigmoid
        + Softmax
        + Softplus
        + Softsign
        + Exponential
        + Linear
    + Trainer (Generic TF2.0 train and validation pipelines) *

Note: `*` - Support coming soon

------------------


## Installation

Before installing `TensorHub`, please install its backend engines: TensorFlow (*TensorFlow 2.0 is Recommended*).

+ [Install TensorFlow and Get Started!](https://www.tensorflow.org/install)

+ [Build, deploy, and experiment easily with TensorFlow](https://www.tensorflow.org/)

You may also consider installing the following **optional dependencies**:

+ [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (*Recommended if you plan on running Keras on GPU*).
+ HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (*Required if you plan on saving Keras models to disk*).

Then, you can install TensorHub itself.

```sh
sudo pip install -U tensorhub
```

If you are using a virtualenv, you may want to avoid using sudo:

```sh
pip install -U tensorhub
```

<!-- + **Alternatively: Install TensorHub from the GitHub source:**

First, clone TensorHub using `git`:

```sh
git clone https://github.com/nityansuman/tensorhub.git
```

Then, `cd` to the TensroHub folder and run the install command:
```sh
cd tensorhub
sudo python setup.py install
``` -->

------------------


## Support

You can also post **bug reports and feature requests** (only) in [GitHub issues](https://github.com/nityansuman/tensorhub/issues). Make sure to read [our guidelines](https://github.com/nityansuman/tensorhub/blob/master/CONTRIBUTING.md) first.

We are eager to collaborate with you. Feel free to open an issue on or send along a pull request.
If you like the work, show your appreciation by "FORK", "STAR", or "SHARE".

[![Love](https://forthebadge.com/images/badges/built-with-love.svg)](https://GitHub.com/nityansuman/tensorhub/)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table>
  <tr>
    <td align="center"><a href="https://www.linkedin.com/in/sanjyot-zade"><img src="https://avatars0.githubusercontent.com/u/14342494?v=4" width="100px;" alt="Sanjyot"/><br /><sub><b>Sanjyot</b></sub></a><br /><a href="https://github.com/nityansuman/tensorhub/commits?author=Sanjyot22" title="Code">💻</a></td>
  </tr>
</table>

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
