# TensorHub <img alt="PyPI" src="https://img.shields.io/pypi/v/tensorhub.svg"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/tensorhub.svg"> <img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/tensorhub.svg"> <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/tensorhub.svg"> <img alt="PyPI - License" src="https://img.shields.io/pypi/l/tensorhub.svg">

The core open source library to help you develop and train ML models easy and fast as never before in TensorFlow.

![TensorHub](data/header.png)

*Next Pre-Release: v1.0beta1 --> 1st July 2019*


## How to use TensorHub

`TensorHub` is a global collection of `Lego blocks` for Neural Networks. You can use it as you like. Only your creativity can stop you from making your own master piece. `TensorHub` gives you the freedom to design your neural architecture / solution and not worry about it"s components.

`TensorHub or THub` for short, is a library of deep learning models and neural lego blocks designed to make deep learning more accessible and accelerate ML research. We provide a set of cooked models that can be used directly with a single call in it"s default configuration or with a custom configuration. We provide a wide range of lego like neural interlocking blocks to so that you can build more and worry less.

Our aim is to provide you enough interlocking building blocks that you can build any neural architecture from basic to advance with less code.


**Tutorials**

+ [Text Classifier Example](examples/run_text_classifiers.py)

*More examples coming soon. Stay put.*


## Available on PyPI
```
pip install tensorhub
```


## Upcoming in v1.0 (More to come... under progress)
```
ROOT
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── data
│   └── header.png
├── examples
│   └── run_text_classifiers.py
├── setup.py
├── tensorhub
│   ├── __init__.py
│   ├── blocks
│   │   ├── attention_wrapper
│   │   │   ├── bahdanau_attention
│   │   │   ├── multi_head_self_attention
│   │   │   └── luong_attention
│   │   ├── layer_wrapper
│   │   │   └── basic_layers
│   │   └── layers (Load wrappers from here)
│   ├── image
│   │   ├── __init__.py
│   │   ├── cooked_models
│   │   │   ├── classification_wrapper
│   │   │   │   ├── ResNet50
│   │   │   │   ├── Xception
│   │   │   │   ├── VGG16
│   │   │   │   ├── VGG19
│   │   │   │   ├── Inceptionv3
│   │   │   │   ├── InceptionResNetv2
│   │   │   │   ├── NasNetLarge
│   │   │   │   ├── NasNetSmall
│   │   │   │   ├── MobileNet
│   │   │   │   ├── Densenet121
│   │   │   │   ├── Densenet169
│   │   │   │   └── Densenet201
│   │   │   └── classifiers (Load classifiers from here)
│   │   └── utilities
│   └── text
│       ├── __init__.py
│       ├── cooked_models
│       │   ├── classification_wrapper
│       │   │   ├── Perceptron
│       │   │   ├── RNN
│       │   │   ├── LSTM
│       │   │   └── GRU
│       │   └── classifiers (Load classifiers from here)
│       └── utilities
│           ├── Trainer
│           │   └── TensorFlow 2.0 Trainer
│           └── processor
│               ├── Load Pre-trained Embeddings
│               └── Create Vocabulary
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


## How To Code in TensorFlow - The Experienced Way

**Sequential Interface**

The best place to start is with the user-friendly Sequential API. You can create simple models by plugging together building blocks. Run the “Hello World” example below, then visit the tutorials to learn more.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile your model
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(data, labels)
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

**Implementing Custom Layers**

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

**Or Like This**
```
class Linear(Layer):
    """y = w.x + b"""

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Instantiate our lazy layer.
linear_layer = Linear(4)

# This will also call `build(input_shape)` and create the weights.
y = linear_layer(tf.ones((2, 2)))
```


## Let's Work Together

```
We're eager to collaborate with you.
Feel free to open an issue on or send along a pull request. Check 'upcoming v1.0' section for new ideas.
```

Drop me an e-mail (nityan.suman@gmail.com) or connect with me on [Linkedin](https://linkedin.com/in/kumar-nityan-suman/) to work together.

If you like the work I do, show your appreciation by "FORK", "STAR", or "SHARE".


[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f34aeed031e249fab2b9df6d74fd1e98)](https://app.codacy.com/app/nityansuman/tensorhub?utm_source=github.com&utm_medium=referral&utm_content=nityansuman/tensorhub&utm_campaign=Badge_Grade_Settings)
[![Love](https://forthebadge.com/images/badges/built-with-love.svg)](https://GitHub.com/nityansuman/tensorhub/)