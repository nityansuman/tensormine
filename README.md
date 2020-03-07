<p align="center"><img src="data/logo.png?raw=true" alt="LOGO"/></p>

<img alt="PyPI" src="https://img.shields.io/pypi/v/tensorhub.svg?color=blue&style=flat"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/tensorhub.svg?style=flat">  <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/nityansuman/tensorhub.svg?color=blue&style=flat"> <img alt="PyPI - License" src="https://img.shields.io/pypi/l/tensorhub.svg?style=flat"> [![Codacy Badge](https://api.codacy.com/project/badge/Grade/d1e35c252db741b28144f5b7b9ffd7d2)](https://www.codacy.com/app/nityansuman/tensorhub?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nityansuman/tensorhub&amp;utm_campaign=Badge_Grade) [![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors)



## You have just found TensorHub.

The open source library to help you develop and train models, easy and fast as never before with `TensorFlow 2.0`.
`TensorHub` is a global hub of `building blocks` and `ready to serve models`.

`TensorHub` is a library of deep learning models and neural lego blocks designed to make deep learning more accessible and accelerate ML research.

Use `TensorHub` if you need a deep learning library that:
+ **Reproducibility** - Reproduce the results of existing pre-training models (such as Google BERT, XLNet).

+ **Model modularity** - TensorHub is divided into multiple components: ready-to-serve models, layers, neural-blocks etc. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules with as few restrictions as possible.

+ **Fast** - Our custom utilities and layers are made from the ground up to support pre-existing standard frameworks like TensorFlow and Keras with efficiency in mind.

+ **Prototyping** - Code less build more. Apply `TensorHub` to create fast prototypes with the help of pre-cooked models, custom layers and utilities support.

+ **Platform Independent** - Supports both `Keras` and `TensorFlow 2.0`. Run your model on CPU, single GPU or using a distributed training strategy.

------------------


## Getting started: 30 seconds to TensorHub

Building a question answering system, an image classification model, a Neural Turing Machine, or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

For a more in-depth tutorial about Keras, you can check out:

+ [Getting started with Text Classification](https://github.com/nityansuman/tensorhub/tree/master/examples/text-classifier-using-tensorhub-models.ipynb)
+ [Getting started with Image Classification](https://github.com/nityansuman/tensorhub/tree/master/examples/image-classifier-using-tensorhub-models.ipynb)
+ [Build your own custom Model](https://github.com/nityansuman/tensorhub/tree/master/examples/working-with-custom-layers.ipynb)

In the [examples folder](https://github.com/nityansuman/tensorhub/tree/master/examples) of this repository, you will find much more advanced examples coming your way very soon.

------------------


## What's coming in V1.0
+ Pre-built Models
    + Image Classification Models (w/ Transfer Learning on ImageNet Weights)
        + Xception
        + VGG16
        + VGG19
        + MobileNet
        + ResNet50
        + InceptionV3
        + InceptionResNetV2
        + Xception
        + DenseNet121
        + DenseNet169
        + DenseNet201
        + NASNetMobile
        + NASNetLarge
        + SmallVGG
        + InceptionV4
        
    + Text Classification Models
        + Basic LSTM/GRU for Sequence Classification Model
        + CNN for Sequence Classification Model
        
    + NMT Models
        + Encoder-Decoder Sequence Translation Model
        + Attention Based Sequence Translation Model

    + Named Entity Recogniton Models
        + Stacked BiLSTM With Word/Characrter Embedding Model


+ Custom Modules/Layers
    + Linear Transformation Layer
    + Inception Layers
    + Attention layers


+ Utilities
    + Processor
        + Learn Tokenizer and Create Vocabulary
        + Load Pre-trained Embeddings
    + Activation Functions
        + GELU
        + RELU
        + SELU
        + ELU
        + Tanh
        + Sigmoid
        + Hard Sigmoid
        + Softmax
        + Softplus
        + Softsign
        + Exponential
        + Linear
        + Mish
        + Swish
    + Model Trainer (generic TF2.0 train and validation pipeline)
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
$ pip install tensorhub

or

$ pip install -U tensorhub
```
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
    <td align="center"><a href="https://github.com/navalchand"><img src="https://avatars0.githubusercontent.com/u/25399517?v=4" width="100px;" alt="Naval Chand"/><br /><sub><b>Naval Chand</b></sub></a><br /><a href="https://github.com/nityansuman/tensorhub/commits?author=navalchand" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nazim1021"><img src="https://avatars0.githubusercontent.com/u/39544613?v=4" width="100px;" alt="Nazim Shaikh"/><br /><sub><b>Nazim Shaikh</b></sub></a><br /><a href="https://github.com/nityansuman/tensorhub/commits?author=nazim1021" title="Code">💻</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/misradiganta/"><img src="https://avatars1.githubusercontent.com/u/34192716?v=4" width="100px;" alt="Diganta Misra"/><br /><sub><b>Diganta Misra</b></sub></a><br /><a href="https://github.com/nityansuman/tensorhub/commits?author=digantamisra98" title="Code">💻</a></td>
  </tr>
</table>

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!