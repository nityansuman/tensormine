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
from tensorhub.utilities.activations import relu, gelu


class LayerA(keras.layers.Layer):
    """Inception V4 block A layer implemented as a feature extraction layer."""

    def __init__(self, activation=relu):
        """Class constructor to initialize variables.

        Keyword Arguments:
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
        """
        super(LayerA, self).__init__()
        self.activation = activation
        self.strides = 1
        self.padding = "same"

    def build(self, input_shape):
        """Lazing building of a layer.

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.conv_1a = keras.layers.Conv2D(96, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1b = keras.layers.Conv2D(96, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1c = keras.layers.Conv2D(64, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_3c = keras.layers.Conv2D(96, (3, 3), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1d = keras.layers.Conv2D(64, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_3d1 = keras.layers.Conv2D(96, (3, 3), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_3d2 = keras.layers.Conv2D(96, (3, 3), activation=self.activation, strides=self.strides, padding=self.padding)
        self.average_pool_a = keras.layers.AveragePooling2D((2, 2), strides=self.strides, padding=self.padding)

    def call(self, x):
        """Forward pass of the layer.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block 1
        out_a_inter = self.average_pool_a(x)
        out_a = self.conv_1a(out_a_inter)
        # Block 2
        out_b = self.conv_1b(x)
        # Block 3
        out_c_inter = self.conv_1c(x)
        out_c = self.conv_3c(out_c_inter)
        # Block 4
        out_d_inter = self.conv_1d(x)
        out_d_inter = self.conv_3d1(out_d_inter)
        out_d = self.conv_3d2(out_d_inter)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b, out_c, out_d], axis=-1)
        return output


class LayerB(keras.layers.Layer):
    """Inception V4 block B layer implemented as a feature extraction layer."""

    def __init__(self, activation=relu):
        """Class constructor to initialize variables.

        Keyword Arguments:
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
        """
        super(LayerB, self).__init__()
        self.activation = activation
        self.strides = 1
        self.padding = "same"

    def build(self, input_shape):
        """Lazing building of a layer.

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.conv_1a = keras.layers.Conv2D(128, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1b = keras.layers.Conv2D(384, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1c = keras.layers.Conv2D(192, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1d = keras.layers.Conv2D(192, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1_7c1 = keras.layers.Conv2D(224, (1, 7), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1_7c2 = keras.layers.Conv2D(256, (1, 7), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1_7d1 = keras.layers.Conv2D(192, (1, 7), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_7_1d1 = keras.layers.Conv2D(224, (7, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1_7d2 = keras.layers.Conv2D(224, (1, 7), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_7_1d2 = keras.layers.Conv2D(256, (7, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.average_pool_a = keras.layers.AveragePooling2D((2, 2), strides=self.strides, padding=self.padding)

    def call(self, x):
        """Forward pass of the layer.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block 1
        out_a_inter = self.average_pool_a(x)
        out_a = self.conv_1a(out_a_inter)
        # Block 2
        out_b = self.conv_1b(x)
        # Block 3
        out_c_inter = self.conv_1c(x)
        out_c_inter = self.conv_1_7c1(out_c_inter)
        out_c = self.conv_1_7c2(out_c_inter)
        # Block 4
        out_d_inter = self.conv_1d(x)
        out_d_inter = self.conv_1_7d1(out_d_inter)
        out_d_inter = self.conv_7_1d1(out_d_inter)
        out_d_inter = self.conv_1_7d2(out_d_inter)
        out_d = self.conv_7_1d2(out_d_inter)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b, out_c, out_d], axis=-1)
        return output


class LayerC(keras.layers.Layer):
    """Inception V4 block C layer implemented as a feature extraction layer."""

    def __init__(self, activation=relu):
        """Class constructor to initialize variables.

        Keyword Arguments:
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
        """
        super(LayerC, self).__init__()
        self.activation = activation
        self.strides = 1
        self.padding = "same"

    def build(self, input_shape):
        """Lazing building of a layer.

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.conv_1a = keras.layers.Conv2D(256, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1b = keras.layers.Conv2D(256, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1c = keras.layers.Conv2D(384, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1d = keras.layers.Conv2D(384, (1, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1_3c = keras.layers.Conv2D(256, (1, 3), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1_3d1 = keras.layers.Conv2D(448, (1, 3), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_1_3d2 = keras.layers.Conv2D(256, (1, 3), activation=self.activation, strides=self.strides,padding=self.padding)
        self.conv_3_1c = keras.layers.Conv2D(256, (3, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_3_1d1 = keras.layers.Conv2D(512, (3, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.conv_3_1d2 = keras.layers.Conv2D(256, (3, 1), activation=self.activation, strides=self.strides, padding=self.padding)
        self.average_pool_a = keras.layers.AveragePooling2D((2, 2), strides=self.strides, padding=self.padding)


    def call(self, x):
        """Forward pass of the layer.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block 1
        out_a_inter = self.average_pool_a(x)
        out_a = self.conv_1a(out_a_inter)
        # Block 2
        out_b = self.conv_1b(x)
        # Block 3
        out_c_inter = self.conv_1c(x)
        out_c1 = self.conv_1_3c(out_c_inter)
        out_c2 = self.conv_3_1c(out_c_inter)
        # Block 4
        out_d_inter = self.conv_1d(x)
        out_d_inter = self.conv_1_3d1(out_d_inter)
        out_d_inter = self.conv_3_1d1(out_d_inter)
        out_d1 = self.conv_1_3d2(out_d_inter)
        out_d2 = self.conv_3_1d2(out_d_inter)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b, out_c1, out_c2, out_d1, out_d2], axis=-1)
        return output


class ReductionLayerA(keras.layers.Layer):
    """Inception V4 Reduction-A layer implemented as a feature extraction layer."""

    def __init__(self, activation=relu):
        """Class constructor to initialize variables.

        Keyword Arguments:
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
        """
        super(ReductionLayerA, self).__init__()
        self.activation = activation
        self.s1 = 1
        self.s2 = 2
        self.pad_same = "same"
        self.pad_valid = "valid"

    def build(self, input_shape):
        """Lazing building of a layer.

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.conv3_b = keras.layers.Conv2D(384, (3, 3), activation=self.activation, strides=self.s2, padding=self.pad_valid)
        self.conv1_c = keras.layers.Conv2D(192, (1, 1), activation=self.activation, strides=self.s1, padding=self.pad_same)
        self.conv3_c1 = keras.layers.Conv2D(224, (3, 3), activation=self.activation, strides=self.s1, padding=self.pad_same)
        self.conv3_c2 = keras.layers.Conv2D(256, (3, 3), activation=self.activation, strides=self.s2, padding=self.pad_valid)
        self.max_pool_a = keras.layers.MaxPooling2D((3, 3), strides=self.s2, padding=self.pad_valid)

    def call(self, x):
        """Forward pass of the layer.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block1
        out_a = self.max_pool_a(x)
        # Block2
        out_b = self.conv3_b(x)
        # Block3
        out_c_inter = self.conv1_c(x)
        out_c_inter = self.conv3_c1(out_c_inter)
        out_c = self.conv3_c2(out_c_inter)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b, out_c], axis=-1)
        return output


class ReductionLayerB(keras.layers.Layer):
    """Inception V4 Reduction-B layer implemented as a feature extraction layer."""

    def __init__(self, activation=relu):
        """Class constructor to initialize variables.

        Keyword Arguments:
            activation {str} -- Activation to be applied on each convolution. (default: {"relu"})
        """
        super(ReductionLayerB, self).__init__()
        self.activation = activation
        self.s1 = 1
        self.s2 = 2
        self.pad_same = "same"
        self.pad_valid = "valid"

    def build(self, input_shape):
        """Lazing building of a layer.

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.max_pool_a = keras.layers.MaxPooling2D((3, 3), strides=self.s2, padding=self.pad_valid)
        self.conv1_b = keras.layers.Conv2D(192, (1, 1), activation=self.activation, strides=self.s1, padding=self.pad_same)
        self.conv3_b = keras.layers.Conv2D(192, (3, 3), activation=self.activation, strides=self.s2, padding=self.pad_valid)
        self.conv1_c = keras.layers.Conv2D(256, (1, 1), activation=self.activation, strides=self.s1, padding=self.pad_same)
        self.conv1_7_c1 = keras.layers.Conv2D(256, (1, 7), activation=self.activation, strides=self.s1, padding=self.pad_same)
        self.conv7_1_c2 = keras.layers.Conv2D(320, (7, 1), activation=self.activation, strides=self.s1, padding=self.pad_same)
        self.conv3_c = keras.layers.Conv2D(320, (3, 3), activation=self.activation, strides=self.s2, padding=self.pad_valid)

    def call(self, x):
        """Forward pass of the layer.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        # Block1
        out_a = self.max_pool_a(x)
        # Block2
        out_b_inter = self.conv1_b(x)
        out_b = self.conv3_b(out_b_inter)
        # Block3
        out_c_inter = self.conv1_c(x)
        out_c_inter = self.conv1_7_c1(out_c_inter)
        out_c_inter = self.conv7_1_c2(out_c_inter)
        out_c = self.conv3_c(out_c_inter)
        # Combine results from each block
        output = keras.layers.concatenate([out_a, out_b, out_c], axis=-1)
        return output