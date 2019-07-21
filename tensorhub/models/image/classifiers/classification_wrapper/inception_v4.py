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
from tensorhub.layers.layer_wrapper.inception_v4 import LayerA
from tensorhub.layers.layer_wrapper.inception_v4 import LayerB
from tensorhub.layers.layer_wrapper.inception_v4 import LayerC
from tensorhub.layers.layer_wrapper.inception_v4 import ReductionLayerA
from tensorhub.layers.layer_wrapper.inception_v4 import ReductionLayerB


class InceptionV4:
    """InceptionV4 is a convolution neural network architecture that is one of the SOTA image classification architectures."""

    def __init__(self, n_classes):
        """Class constructor.

        Arguments:
            n_classes {int} -- Number of classes for classification.

        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
        """
        self.n_classes = n_classes


    def model(self):
        """Creates InceptionV4 CNN architecture.
        
        Returns:
            keras-model -- Build InceptionV4 model with inceptionv4 layer A,B,C and inception reduction layer A & B.
        """
        input_t = keras.layers.Input(shape=(299, 299, 3))
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation="relu", strides=2)(input_t)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", strides=1)(x)

        x1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(x)
        x2 = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding="valid", activation="relu", strides=2)(x)
        x = keras.layers.concatenate([x1, x2], axis=-1)

        x1 = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding="same", activation="relu", strides=1)(x)
        x1 = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(x1)

        x2 = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding="same", activation="relu", strides=1)(x)
        x2 = keras.layers.Conv2D(filters=64, kernel_size=(1, 7), padding="same", activation="relu", strides=1)(x2)
        x2 = keras.layers.Conv2D(filters=64, kernel_size=(7, 1), padding="same", activation="relu", strides=1)(x2)
        x2 = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding="valid", activation="relu", strides=1)(x2)
        x = keras.layers.concatenate([x1, x2], axis=-1)

        x1 = keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding="valid", activation="relu", strides=2)(x)
        x2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)
        x = keras.layers.concatenate([x1, x2], axis=-1)

        for i in range(4):
            x = LayerA()(x)
        x = ReductionLayerA()(x)
        for i in range(7):
            x = LayerB()(x)
        x = ReductionLayerB()(x)
        for i in range(3):
            x = LayerC()(x)

        x = keras.layers.AveragePooling2D(pool_size=(8, 8))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(rate=0.2)(x)
        x = keras.layers.Dense(units=self.n_classes, activation="softmax")(x)
        return keras.models.Model(input_t, x)