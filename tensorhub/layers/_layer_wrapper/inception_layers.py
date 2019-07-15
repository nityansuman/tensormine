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


class InceptionV1(keras.layers.Layer):
    """Inception V1 module implemented as a feature extraction layer."""

    def __init__(self, num_filters=28, activation="relu"):
        """Class constructor to initialize variables.
        
        Keyword Arguments:
            num_filters {int} -- Number of filters for convolution. (default: {28})
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
        """
        super(InceptionV1, self).__init__()
        self.num_filters = num_filters
        self.activation = activation
        self.strides = 1
        self.padding = "same"

    def build(self, input_shape):
        """Lazing building of a layer.
        
        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.conv_block_a = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_block_b = keras.layers.Conv2D(self.num_filters, (3, 3), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_block_c = keras.layers.Conv2D(self.num_filters, (5, 5), activation=self.activation, strides=self.strides, padding=self.padding)
        self.maxpool_block = keras.layers.MaxPool2D(pool_size=(3, 3), strides=self.strides, padding=self.padding)
    
    def call(self, x):
        """Forward pass of the layer.
        
        Arguments:
            x {tensor} -- Input tensor to the layer.
        
        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block 1
        out_a = self.conv_block_a(x)
        # Block 2
        out_b = self.conv_block_b(x)
        # Block 3
        out_c = self.conv_block_a(x)
        # Block 4
        out_d = self.maxpool_block(x)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b, out_c, out_d], axis=-1)
        return output

class InceptionV1Reduction(keras.layers.Layer):
    """Inception V1 with reduction module implemented as a feature extraction layer."""

    def __init__(self, num_filters=28, activation="relu"):
        """Class constructor to initialize variables.
        
        Keyword Arguments:
            num_filters {int} -- Number of filters for convolution. (default: {28})
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
        """
        super(InceptionV1, self).__init__()
        self.num_filters = num_filters
        self.activation = activation
        self.strides = 1
        self.padding = "same"

    def build(self, input_shape):
        """Lazing building of a layer.
        
        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.conv_1a = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=self.strides*self.strides, padding=self.padding)
        self.conv_1b = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1c = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1d = keras.layers.Conv2D(self.num_filters, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_3 = keras.layers.Conv2D(self.num_filters, (3, 3), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_5 = keras.layers.Conv2D(self.num_filters, (5, 5), activation=self.activation, strides=self.strides, padding=self.padding)
        self.maxpool_block = keras.layers.MaxPool2D(pool_size=(3, 3), strides=self.strides, padding=self.padding)
    
    def call(self, x):
        """Forward pass of the layer.
        
        Arguments:
            x {tensor} -- Input tensor to the layer.
        
        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block 1
        out_a = self.conv_1a(x)
        # Block 2
        out_b_inter = self.conv_1b(x)
        out_b = self.conv_3(out_b_inter)
        # Block 3
        out_c_inter = self.conv_1c(x)
        out_c = self.conv_5(out_c_inter)
        # Block 4
        out_d_inter = self.maxpool_block(x)
        out_d = self.conv_1d(out_d_inter)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b, out_c, out_d], axis=-1)
        return output