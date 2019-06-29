# TensorHub <img alt="PyPI" src="https://img.shields.io/pypi/v/tensorhub.svg"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/tensorhub.svg"> <img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/tensorhub.svg"> <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/tensorhub.svg"> <img alt="PyPI - License" src="https://img.shields.io/pypi/l/tensorhub.svg">

The open source library to help you develop and train ML models easy and fast as never before in TensorFlow.

![TensorHub](data/header.png)

*Next Pre-Release: v1.0alpha3 --> 1st July 2019*

## Available on PyPI
```
pip install tensorhub
```

## How to use TensorHub

`TensorHub` is a global collection of `blocks` and `ready to serve models`. You can use it as you like. Only your creativity can stop you from making your own master piece. `TensorHub` gives you the freedom to design your neural architecture / solution and not worry about it"s components.

`TensorHub or THub` for short, is a wrapper library of deep learning models and neural lego blocks designed to make deep learning more accessible and accelerate ML research with `TensorFlow 2.0`. 

Our aim is to provide you enough interlocking building blocks that you can build any neural architecture from basic to advance with less code.


**Tutorials**

+ [Learn TensorFlow 2.0](examples/)
+ [Text Classifier Example](examples/run_text_classifiers.py)

`More examples coming soon. Stay put.`


## Coming in TensorHub v1.0
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

    + Text Classification
        + RNN, LSTM, GRU Based Model
        + Text-CNN

    + Neural Machine Translation
        + Encoder-Decoder Standard Sequence Model (Encoder, Decoder -> [RNN, LSTM, GRU])
        + Translation with Attention

    + Text Generation
        + RNN, LSTM, GRU Based Model
    
    + Named Entity Recogniton
        + RNN, LSTM, GRU Based Model

+ Layers
    + Standard
        + Linear
        + *Under planning*
    + Attention
        + Bahdanau Attention
        + Luong Attention
        + Self-Attention
        + *More to come...*


+ Losses
    + Mean Squared Error
    + Mean Absolute Error
    + Mean Absolute Percentage Error
    + Mean Squared Algorithmic Error
    + Hige
    + Squared Hinge
    + Binary Cross Entropy
    + Categorical Cross Entropy
    + Sparse Categorical Cross Entropy
    + Poisson
    
+ Metrics
    + Binary Accuracy
    + Categorical Accuracy
    + Sparse Categorcial Accuracy
    + Top-k Categorical Accuracy
    + Sparse Top-K Categorical Accuracy
    + Mean Prediction
    + Precision
    + Recall/Sensitivity
    + F1-Score
    + Matthews correlation coefficient (MCC)
    + Specificity/Selectivity 
    + False Discovery Rate
    + False Omission Rate

    (*All `Losses` can be used as a metric to evaluate the model performance.*)

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

+ Utilities
    + Text
        + Custom Tokenizer
        + Load Embeddings
        + *More to come...*
    + Image
        + *More to come...*


## Why TensorFlow
TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

**Easy model building**
Build and train ML models easily using intuitive high-level APIs like Keras with eager execution, which makes for immediate model iteration and easy debugging.

**Robust ML production anywhere**
Easily train and deploy models in the cloud, on-prem, in the browser, or on-device no matter what language you use.

**Powerful experimentation for research**
A simple and flexible architecture to take new ideas from concept to code, to state-of-the-art models, and to publication faster.

**[Install TensorFlow and Get Started!](https://www.tensorflow.org/install)**

**[Build, deploy, and experiment easily with TensorFlow](https://www.tensorflow.org/)**


## Let's Work Together

```
We're eager to collaborate with you.
Feel free to open an issue on or send along a pull request.
```

Drop me an e-mail (nityan.suman@gmail.com) or connect with me on [Linkedin](https://linkedin.com/in/kumar-nityan-suman/) to work together.

If you like the work I do, show your appreciation by "FORK", "STAR", or "SHARE".


[![Love](https://forthebadge.com/images/badges/built-with-love.svg)](https://GitHub.com/nityansuman/tensorhub/)