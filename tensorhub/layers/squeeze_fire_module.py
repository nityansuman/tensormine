# Load packages
from tensorflow import keras


class SqueezeFireModule(keras.layers.Layer):
    """Inception V1 module implemented as a feature extraction layer."""

    def __init__(self, fire_nodes=None, skip_connection=False, name=None):
        """Class constructor to initialize variables.

        Keyword Arguments:
            num_filters {int} -- Number of filters for convolution. (default: {28})
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
        """
        if name:
            super(SqueezeFireModule, self).__init__(name=name)
        else:
            super(SqueezeFireModule, self).__init__()
        
        self.fire_nodes = [128, 256, 512, 1024] if fire_nodes == None else fire_nodes
        self.skip_connection = skip_connection

    def build(self, input_shape):
        """Lazing building of a layer.

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.fire_conv = keras.layers.Conv2D(filters=int(self.fire_nodes[0]), kernel_size=(1, 1), strides=(1, 1), activation=None, padding="same")
        self.fire_batch = keras.layers.BatchNormalization()
        self.fire_relu = keras.layers.Activation("relu")
        self.fire_conv_1 = keras.layers.Conv2D(filters=int(self.fire_nodes[1]), kernel_size=(1, 1), strides=(1, 1),activation=None, padding="same")
        self.fire_batch_1 = keras.layers.BatchNormalization()
        self.fire_conv_2 = keras.layers.Conv2D(filters=int(self.fire_nodes[2]), kernel_size=(3, 3), strides=(1, 1), activation=None, padding="same")
        self.fire_batch_2 = keras.layers.BatchNormalization()
        self.fire_penaltimate = keras.layers.Concatenate(axis=-1)
        self.skip_connection_output = keras.layers.Add()
        self.final_relu = keras.layers.Activation("relu")
        self.fire_pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")


    def call(self, x):
        """Forward pass of the layer.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        x_ = x
        x = self.fire_conv(x)
        x = self.fire_batch(x)
        x = self.fire_relu(x)

        x1 = self.fire_conv_1(x)
        x1 = self.fire_batch_1(x1)

        x2 = self.fire_conv_2(x)
        x2 = self.fire_batch_2(x2)

        x = self.fire_penaltimate([x1, x2])

        if self.skip_connection:
            x = self.skip_connection_output()([x, x_])
            return self.fire_pool(self.final_relu(x))
        else:
            return self.final_relu(x)