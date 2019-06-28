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
