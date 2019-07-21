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
from tensorhub.utilities.activations import relu, softmax
from tensorhub.layers.layer_wrapper.inception_v4 import LayerA
from tensorhub.layers.layer_wrapper.inception_v4 import LayerB
from tensorhub.layers.layer_wrapper.inception_v4 import LayerC
from tensorhub.layers.layer_wrapper.inception_v4 import ReductionLayerA
from tensorhub.layers.layer_wrapper.inception_v4 import ReductionLayerB


class InceptionV4(keras.models.Model):
    """InceptionV4 is a convolution neural network architecture that is one of the SOTA image classification architectures."""

    def __init__(self, num_classes, act=relu, output_act=softmax):
        """Class constructor.

        Arguments:
            num_classes {int} -- Number of class labels.

        Keyword Arguments:
            act {str/tensorhub.utilities.activation} -- Activation to be used with hidden layers of the network. (default: {'relu'})
            output_act {str/tensorhub.utilities.activation} -- Activation to be used with output layer. (default: {'softmax'})
        """
        super(InceptionV4, self).__init__()
        # Se member variables
        self.num_classes = num_classes
        self.act = act
        self.output_act = output_act
        # Define layers
        # Stem block 1
        self.conv_1a = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation=self.act)
        self.conv_1b = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation=self.act)
        self.conv_1c = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=self.act)
        self.conv_1d = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding="valid")
        self.max_pool_layer_1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")
        # Stem block 2
        self.conv_2a = keras.layers.Conv2D(filters=64, kernel_size=(1, 1))
        self.conv_2b = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding="valid")
        self.conv_2c = keras.layers.Conv2D(filters=64, kernel_size=(1, 1))
        self.conv_2d = keras.layers.Conv2D(filters=64, kernel_size=(7, 1))
        self.conv_2e = keras.layers.Conv2D(filters=64, kernel_size=(1, 7))
        self.conv_2f = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding="valid")
        # Stem block 3
        self.conv_3a = keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding="valid")
        self.max_pool_layer_3 = keras.layers.MaxPool2D(strides=(2, 2), padding="valid")
        self.concat_layer = keras.layers.concatenate(axis=-1)
        # Inception layer A
        self.inception_layer_a1 = LayerA()
        self.inception_layer_a2 = LayerA()
        self.inception_layer_a3 = LayerA()
        self.inception_layer_a4 = LayerA()
        self.reduction_a = ReductionLayerA()
        # Inception layer B
        self.inception_layer_b1 = LayerB()
        self.inception_layer_b2 = LayerB()
        self.inception_layer_b3 = LayerB()
        self.inception_layer_b4 = LayerB()
        self.inception_layer_b5 = LayerB()
        self.inception_layer_b6 = LayerB()
        self.inception_layer_b7 = LayerB()
        self.reduction_b = ReductionLayerB()
        # Inception layer C
        self.inception_layer_c1 = LayerC()
        self.inception_layer_c2 = LayerC()
        self.inception_layer_c3 = LayerC()
        # Average pooling
        self.avg_pool_layer_1 = keras.layers.AveragePooling2D()
        self.dropout_layer = keras.layers.Dropout(rate=0.2)
        # Softmax
        self.d1 = keras.layers.Dense(units=2048, activation=relu)
        self.d2 = keras.layers.Dense(units=self.num_classes, activation=self.output_act)

    def call(self, x):
        # Inception v4 stem
        # Stem block 1
        x = self.conv_1a(x)
        x = self.conv_1b(x)
        x = self.conv_1c(x)
        x1 = self.max_pool_layer(x)
        x2 = self.conv_1d(x)
        x = self.concat_layer([x1, x2])
        # Stem block 2
        x1 = self.conv_2a(x)
        x1 = self.conv_2b(x1)
        x2 = self.conv_2c(x)
        x2 = self.conv_2d(x2)
        x2 = self.conv_2e(x2)
        x2 = self.conv_2f(x2)
        x = self.concat_layer([x1, x2])
        # Stem block 3
        x1 = self.conv_3a(x)
        x2 = self.max_pool_layer_3(x)
        x = self.concat_layer([x1, x2])
        # Inception layer A
        x = self.inception_layer_a1(x)
        x = self.inception_layer_a2(x)
        x = self.inception_layer_a3(x)
        x = self.inception_layer_a4(x)
        x = self.reduction_a(x)
        # Inception layer B
        x = self.inception_layer_b1(x)
        x = self.inception_layer_b2(x)
        x = self.inception_layer_b3(x)
        x = self.inception_layer_b4(x)
        x = self.inception_layer_b5(x)
        x = self.inception_layer_b6(x)
        x = self.inception_layer_b7(x)
        x = self.reduction_b(x)
        # Inception layer C
        x = self.inception_layer_c1(x)
        x = self.inception_layer_c2(x)
        x = self.inception_layer_c3(x)
        # Tail
        x = self.avg_pool_layer_1(x)
        x = self.dropout_layer(x)
        # Softmax
        x = self.d1(x)
        x = self.d2(x)
        return x