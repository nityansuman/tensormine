<p align="center"><img src="metadata/th-logo.png?raw=true" alt="LOGO"/></p>

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/92d9bc37335c4fda8bedb50455ef1233)](https://app.codacy.com/manual/nityansuman/tensorhub?utm_source=github.com&utm_medium=referral&utm_content=nityansuman/tensorhub&utm_campaign=Badge_Grade_Settings)
![GitHub LICENSE](https://img.shields.io/github/license/nityansuman/tensorhub)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/nityansuman/tensorhub)
![GitHub repo size](https://img.shields.io/github/repo-size/nityansuman/tensorhub)
![GitHub language count](https://img.shields.io/github/languages/count/nityansuman/tensorhub)
![GitHub last commit](https://img.shields.io/github/last-commit/nityansuman/tensorhub)

## You have just found TensorHub!

TensorHub is a deep learning API written in Python, running on top of the machine learning platform TensorFlow 2 to provide simple, modular and repeatable abstractions to accelerate deep learning research. TensorHub is designed to be simple to understand, easy to write and quick to change.

Unlike many frameworks TensorHub is extremely flexible about how to use modules. Modules are designed to be self contained and entirely decoupled from one another.

Use TensorHub if you need a deep learning library that:

- **Reproducibility** - Reproduce the results of existing pre-training models (such as ResNet, VGG, BERT, XLNet).

- **Modularity** - Clear and robust interface allows users to combine modules with as few restrictions as possible.

- **Fast** - Our custom utilities and layers are made from the ground up to support pre-existing standard frameworks like TensorFlow and Keras with efficiency in mind.

- **Prototyping** - Code less build more. Apply modular blocks to create fast prototypes with the help of pre-cooked models, custom layers and utilities support.

- **Platform Independent** - Run your model on CPU, single GPU or using a distributed training strategy on top of TensorFlow 2.

## Installation & Compatibility

To use, simply install from [PyPI](https://pypi.org/) via `pip`:

```
$ pip install tensorhub
```

TensorHub is compatible with:
- Python 3.5–3.8
- TensorFlow 2.3.0 or later
- Ubuntu 16.04 or later
- Windows 7 or later
- macOS 10.12.6 (Sierra) or later.

## Getting Started

The ideas behind deep learning are simple, so why should their implementation be painful?

TensorHub ships with a number of built in modules like pre-built `models` and advance `layers` that can be used easily.

### Models on a Plate (MoaP)

MoaP's are deep learning models that are made available with TensorHub. These models can be used for training, feature extraction, fine-tuning or as you wish.

### Layers

Layers are the basic building blocks of neural networks in TensorHub. A layer consists of a tensor-in tensor-out computation function (the layer's call method) and some state, held in TensorFlow variables (the layer's weights).

TensorHub provides customs layers conceptualized from proven and high performing deep learning models. This helps to take advantage of core magic blocks from high performing SOTA models with smaller or a different neural architecture.

## Support

You can also post bug reports and feature requests (only) in GitHub issues. Make sure to read our guidelines first.

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Love](https://forthebadge.com/images/badges/built-with-love.svg)](https://GitHub.com/nityansuman/tensorhub/)
