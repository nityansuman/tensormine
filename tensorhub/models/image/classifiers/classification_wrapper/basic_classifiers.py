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
from tensorhub.utilities.activations import relu, softmax, sigmoid


class SmallVGG(keras.models.Model):
    """Small VGG like image classifier."""

    def __init__(self, num_classes, dense_layer_size=1024, activation=relu, output_activation=softmax, dp_rate=0.25):
        """Class constructor to initialise member variables.
        
        Arguments:
            num_classes {[type]} -- Number of prediction classes.
        
        Keyword Arguments:
            dense_layer_size {int} -- Number of nodes in the dense layer. (default: {1024})
            activation {str} -- Activation to be used in the hidden layers. (default: {relu})
            output_activation {str} -- Activation to be used with the output layer. (default: {softmax})
            dp_rate {float} -- Dropout ratio to be used. (default: {0.25})
        """
        super(CNNClassifier, self).__init__()
        # Set member variables
        self.num_classes = num_classes
        self.dense_layer_size = dense_layer_size
        self.activation = activation
        self.output_activation = output_activation
        self.dp_rate = dp_rate
        # Define layers
        self.conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=self.activation)
        self.conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=self.activation)
        self.conv3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=self.activation)
        self.conv4 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=self.activation)
        self.pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))
        self.pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))
        self.dropout = keras.layers.Dropout(rate=dp_rate)
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(units=self.dense_layer_size, activation=self.activation)
        self.d2 = keras.layers.Dense(units=self.dense_layer_size, activation=self.activation)
        self.d3 = keras.layers.Dense(units=self.num_classes, activation=self.output_activation)


    def call(self, x):
        """Forward pass over the network.
        
        Returns:
            output -- Output tensor from the network.
        """
        # First convolution block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Second convolution block
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.dropout(x)

        # Classification layer
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout(x)
        x = self.d2(x)
        x = self.d3(x)
        return x