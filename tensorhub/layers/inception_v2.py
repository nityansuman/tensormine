# Copyright 2019 The TensorHub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Load packages
from tensorflow import keras


class LayerA(keras.layers.Layer):
    """Standard Inception V2 (Inception V2 module A) building block implemented as a layer for advance feature creation on images.

    Know more at: https://arxiv.org/pdf/1512.00567v3.pdf
    """

    def __init__(self, num_filters=768, activation="relu", name="inception_v2_layer_a"):
        """Initialize variables.

        Keyword Arguments:
            num_filters {int} -- Number of filters for convolution. (default: {288})
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
            name {str} -- Name associated with this layer. (default: {None})
        """
        super(LayerA, self).__init__(name=name)
        self.num_filters = num_filters
        self.activation = activation

    def build(self, input_shape):
        """The __call__ method of your layer will automatically run build the first time it is called.
        You now have a layer that's lazy and easy to use.

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.conv_1a = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1b = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1c = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1d = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_3b = keras.layers.Conv2D(self.num_filters, (3, 3), activation=self.activation, strides=1, padding="same")
        self.conv_3c1 = keras.layers.Conv2D(self.num_filters, (3, 3), activation=self.activation, strides=1, padding="same")
        self.conv_3c2 = keras.layers.Conv2D(self.num_filters, (3, 3), activation=self.activation, strides=1, padding="same")
        self.maxpool_layer = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="same")

    def call(self, x):
        """Forward pass over the layer.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block 1
        out_a = self.conv_1a(x)
        # Block 2
        out_b_inter = self.conv_1b(x)
        out_b = self.conv_3b(out_b_inter)
        # Block 3
        out_c_inter = self.conv_1c(x)
        out_c_inter = self.conv_3c1(out_c_inter)
        out_c = self.conv_3c2(out_c_inter)
        # Block 4
        out_d_inter = self.maxpool_layer(x)
        out_d = self.conv_1d(out_d_inter)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b, out_c, out_d], axis=-1)
        return output


class LayerB(keras.layers.Layer):
    """Inception V2 module B building block implemented as a layer for advance feature creation on images.

    Know more at: https://arxiv.org/pdf/1512.00567v3.pdf
    """

    def __init__(self, num_filters=1280, activation="relu", name="inception_v2_layer_b"):
        """Initialize variables.

        Keyword Arguments:
            num_filters {int} -- Number of filters for convolution. (default: {1280})
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
            name {str} -- Name associated with this layer. (default: {None})
        """
        super(LayerB, self).__init__(name=name)
        self.num_filters = num_filters
        self.activation = activation

    def build(self, input_shape):
        """The __call__ method of your layer will automatically run build the first time it is called.
        You now have a layer that's lazy and easy to use

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.conv_1a = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1b = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1_3b = keras.layers.Conv2D(self.num_filters, (1, 3), activation=self.activation, strides=1, padding="same")
        self.conv_3_1b = keras.layers.Conv2D(self.num_filters, (3, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1c = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1_3c1 = keras.layers.Conv2D(self.num_filters, (1, 3), activation=self.activation, strides=1, padding="same")
        self.conv_3_1c1 = keras.layers.Conv2D(self.num_filters, (3, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1_3c2 = keras.layers.Conv2D(self.num_filters, (1, 3), activation=self.activation, strides=1, padding="same")
        self.conv_3_1c2 = keras.layers.Conv2D(self.num_filters, (3, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1d = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.maxpool_layer = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="same")

    def call(self, x):
        """Forward pass over the layer.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block 1
        out_a = self.conv_1a(x)
        # Block 2
        out_b_inter = self.conv_1b(x)
        out_b_inter = self.conv_1_3b(out_b_inter)
        out_b = self.conv_3_1b(out_b_inter)
        # Block 3
        out_c_inter = self.conv_1c(x)
        out_c_inter = self.conv_1_3c1(out_c_inter)
        out_c_inter = self.conv_3_1c1(out_c_inter)
        out_c_inter = self.conv_1_3c2(out_c_inter)
        out_c = self.conv_3_1c2(out_c_inter)
        # Block 4
        out_d_inter = self.maxpool_layer(x)
        out_d = self.conv_1d(out_d_inter)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b, out_c, out_d], axis=-1)
        return output


class LayerC(keras.layers.Layer):
    """Inception V2 module C building block implemented as a layer for advance feature creation on images.

    Know more at: https://arxiv.org/pdf/1512.00567v3.pdf
    """

    def __init__(self, num_filters=2048, activation="relu", name="inception_v2_layer_c"):
        """Initialize variables.

        Keyword Arguments:
            num_filters {int} -- Number of filters for convolution. (default: {2048})
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
            name {str} -- Name associated with this layer. (default: {None})
        """
        super(LayerC, self).__init__(name=name)
        self.num_filters = num_filters
        self.activation = activation

    def build(self, input_shape):
        """The __call__ method of your layer will automatically run build the first time it is called.
        You now have a layer that's lazy and easy to use

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.conv_1a = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1b = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1_3b = keras.layers.Conv2D(self.num_filters, (1, 3), activation=self.activation, strides=1, padding="same")
        self.conv_3_1b = keras.layers.Conv2D(self.num_filters, (3, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1c = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.conv_3c = keras.layers.Conv2D(self.num_filters, (3, 3), activation=self.activation,strides=1, padding="same")
        self.conv_1_3c = keras.layers.Conv2D(self.num_filters, (1, 3), activation=self.activation, strides=1, padding="same")
        self.conv_3_1c = keras.layers.Conv2D(self.num_filters, (3, 1), activation=self.activation, strides=1, padding="same")
        self.conv_1d = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=1, padding="same")
        self.maxpool_layer = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="same")

    def call(self, x):
        """Forward pass over the layer.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block 1
        out_a = self.conv_1a(x)
        # Block 2
        out_b_inter = self.conv_1b(x)
        out_b1 = self.conv_1_3b(out_b_inter)
        out_b2 = self.conv_3_1b(out_b_inter)
        # Block 3
        out_c_inter = self.conv_1c(x)
        out_c_inter = self.conv_3c(out_c_inter)
        out_c1 = self.conv_1_3c(out_c_inter)
        out_c2 = self.conv_3_1c(out_c_inter)
        # Block 4
        out_d_inter = self.maxpool_layer(x)
        out_d = self.conv_1d(out_d_inter)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b1, out_b2, out_c1, out_c2, out_d], axis=-1)
        return output