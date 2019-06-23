# TensorHub <img alt="PyPI" src="https://img.shields.io/pypi/v/tensorhub.svg"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/tensorhub.svg"> <img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/tensorhub.svg"> <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/tensorhub.svg"> <img alt="PyPI - License" src="https://img.shields.io/pypi/l/tensorhub.svg">

**Models Implemented in TensorFlow 2.0** (*Under active development.*)

The core open source library to help you develop and train ML models easy and fast as never before.

![TensorHub](data/header.png)

## How to use TensorHub

`TensorHub` is a global collection of `Lego blocks` for Neural Networks. You can use it as you like. Only your creativity can stop you from making your own master piece. `TensorHub` gives you the freedom to design your neural architecture / solution and not worry about it"s components.

Our aim is to provide you enough interlocking building blocks that you can build any neural architecture from basic to advance in less than `15 minutes` with less than `30 lines` of codes in `TensorFlow 2.0`.

`TensorHub or THub` for short, is a library of deep learning models and neural lego blocks designed to make deep learning more accessible and accelerate ML research. We provide a set of cooked models that can be used directly with a single call in it"s default configuration or with a custom configuration. We provide a wide range of lego like neural interlocking blocks to so that you can build more and worry less.


## Requirements
+ Tensorflow 2.0.0-beta1
+ Python 3.0

## Install
```
pip install tensorhub
```

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


## How To Code in TensorFlow 2.0

**Sequential Interface**

The best place to start is with the user-friendly Sequential API. You can create models by plugging together building blocks. Run the “Hello World” example below, then visit the tutorials to learn more.

```
my_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

OR

inputs = tf.keras.layers.Input(shape=(784,))
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)
my_model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
```

**Functional Interface**

```
# Input to the model
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation="relu")(inputs)
x = Dense(64, activation="relu")(x)

# Ouptut of the model
predictions = Dense(10, activation="softmax")(x)

# This creates a model that includes the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

# Compile your model
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(data, labels)
```

**Subclassing Interface**

The Subclassing API provides a define-by-run interface for advanced research. Create a class for your model, then write the forward pass imperatively. Easily **author custom layers**, **activations**, **training loop** and much more.

```
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Call your model
model = MyModel()
```

**Implementing custom layers**

The best way to implement your own layer is extending the tf.keras.Layer class and implementing: * __init__ , where you can do all input-independent initialization * build, where you know the shapes of the input tensors and can do the rest of the initialization * call, where you do the forward computation

```
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_variable("kernel", shape=[int(input_shape[-1]), self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

# Call your layer
layer = MyDenseLayer(10)
```

## How to use TensorHub

+ [Text Classifier Example](examples/run_text_classification.py)

*Lot more coming your way. Stay put.*


## Whats cooking Inside

+ Natural Language Processing
    + Cooked Models:
        + Text Classification
            + RNN
            + LSTM
            + GRU
        + Neural Machine Translation (Optional Attention)
            + GRU
            + LSTM
    + Building Blocks:
        + Attention
            + Multi-Head Self-Attention
            + Bahdanau Attention
            + Luong Attention
        + Encoder (Optional Attention)
            + RNN
            + GRU
            + LSTM
        + Decoder (Optional Attention)
            + RNN
            + GRU
            + LSTM


```
We"re eager to collaborate with you too,
so feel free to open an issue on GitHub or send along a pull request.
```

## Let's Work Together

Drop me an e-mail (nityan.suman@gmail.com) or connect with me on [Linkedin](https://linkedin.com/in/kumar-nityan-suman/) to work together on an awesome project.

If you like the work I do, show your appreciation by "FORK", "STAR", or "SHARE".
