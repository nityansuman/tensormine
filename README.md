<p align="center"><img src="data/logo.png?raw=true" alt="LOGO"/></p>

<img alt="PyPI" src="https://img.shields.io/pypi/v/tensorhub.svg?color=blue&style=flat"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/tensorhub.svg?style=flat">  <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/nityansuman/tensorhub.svg?color=blue&style=flat"> <img alt="PyPI - License" src="https://img.shields.io/pypi/l/tensorhub.svg?style=flat"> [![Codacy Badge](https://api.codacy.com/project/badge/Grade/d1e35c252db741b28144f5b7b9ffd7d2)](https://www.codacy.com/app/nityansuman/tensorhub?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nityansuman/tensorhub&amp;utm_campaign=Badge_Grade)


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

**Here is the `Sequential` model for `Image Classification`:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense

# Use Tensorhub to accelerate your prototyping
from tensorhub.layers import InceptionV1 # Custom Inception Layer
from tensorhub.models.image.classifiers import CNNClassifier, VGG16 # Cooked Models
from tensorhub.utilities.activations import gelu, softmax # Custom Activations Supported

## Initiate a sequential model
model = Sequential()

## Stacking layers is as easy as `.add()`
model.add(Conv2D(32, (3, 3), activation=gelu, input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation=gelu))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

## Add custom layer like any other standard layer
model.add(InceptionV1(32)) 

model.add(Dense(units=64, activation=gelu, input_dim=100))
model.add(Dense(units=10, activation=softmax))

# Or
## Use one of our pre-cooked models
model = VGG16(n_classes=10, num_nodes=64, img_height=100, img_width=100)

## Once your model looks good, configure its learning process with `.compile()`
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

## Alternatively, if you need to, you can further configure your compile configuration
model.compile(
    loss=tensorflow.keras.losses.categorical_crossentropy,
    optimizer=tensorflow.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
    metrics=['acc']
)

## You can now iterate on your training data in batches

## x_train and y_train are Numpy arrays
model.fit(x_train, y_train, epochs=5, batch_size=32)

## Alternatively, you can feed batches to your model manually
model.train_on_batch(x_batch, y_batch)

## Evaluate your performance in one line:
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

## Or generate predictions on new data
classes = model.predict(x_test, batch_size=128)
```

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

<p align="center"><img src="data/pip_install.png?raw=true" alt="PIP-INSTALL"/></p>

```sh
sudo pip install tensorhub
```

If you are using a virtualenv, you may want to avoid using sudo:

```sh
pip install tensorhub
```

+ **Alternatively: Install TensorHub from the GitHub source:**

First, clone TensorHub using `git`:

```sh
git clone https://github.com/nityansuman/tensorhub.git
```

Then, `cd` to the TensroHub folder and run the install command:
```sh
cd tensorhub
sudo python setup.py install
```

------------------


## Support

You can also post **bug reports and feature requests** (only) in [GitHub issues](https://github.com/nityansuman/tensorhub/issues). Make sure to read [our guidelines](https://github.com/nityansuman/tensorhub/blob/master/CONTRIBUTING.md) first.

We are eager to collaborate with you. Feel free to open an issue on or send along a pull request.
If you like the work, show your appreciation by "FORK", "STAR", or "SHARE".

[![Love](https://forthebadge.com/images/badges/built-with-love.svg)](https://GitHub.com/nityansuman/tensorhub/)
