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


class CNNClassifier:
    """CNN based standard image classifier."""

    def __init__(self, img_shape, num_classes, kernal_sz=None, num_filters=None, pool_sz=None, strides_sz=None, num_cnn_blocks=2, dense_layer_size=1024, activation="relu", data_format="channels_last"):
        """Class constructor to initialise member variables.
        
        Arguments:
            img_shape {tuple} -- Tuple containing image height, width, channels. Channel last format.
            num_classes {[type]} -- Number of prediction classes.
        
        Keyword Arguments:
            kernal_sz {list} -- List of convolution size for each convolutional block. (default: {None})
            num_filters {list} -- List of number of filters for each convolutional block. (default: {None})
            pool_sz {list} -- List of pooling size for each convolutional block. (default: {None})
            strides {list} -- List of values for strides for each convolutional block. (default: {None})
            dense_layer_size {int} -- Number of nodes in the dense layer. (default: {1024})
            num_cnn_blocks {int} -- Number of stacked cnn blocks. (default: {2})
            activation {str} -- Activation for the dense layers. (default: {'relu'})
            data_format {str} -- Image data format. (default: {'channels_last'})
        """
        self.img_shape = img_shape
        self.num_classes = num_classes
        # Set output activation based on the number of classes
        if self.num_classes == 1:
            self.output_activation = "sigmoid"
        else:
            self.output_activation = "softmax"
        self.num_cnn_blocks = num_cnn_blocks
        self.pool_sz = pool_sz if pool_sz != None else [2]*(self.num_cnn_blocks)
        self.num_filters = num_filters if num_filters != None else [32]*(self.num_cnn_blocks)
        self.kernal_sz = kernal_sz if kernal_sz != None else [2]*(self.num_cnn_blocks)
        self.strides_sz = strides_sz if strides_sz != None else [1, 1]*(self.num_cnn_blocks)
        # Assertion check
        if len(self.pool_sz) != self.num_cnn_blocks or len(self.num_filters) != self.num_cnn_blocks or len(self.kernal_sz) != self.num_cnn_blocks:
            raise AssertionError(AssertionError("Assertion Error: `pool_sz`: {}, `num_filters`: {}, `kernal_sz`: {} should be same as `kernal_sz`: {}".format(len(self.pool_sz), len(self.num_filters), len(self.kernal_sz), self.num_cnn_blocks)))
        self.activation = activation
        self.dense_layer_size = dense_layer_size

    def model(self):
        """Create `CNN` based image classifier.
        
        Returns:
            keras.models.Sequential -- Instance of keras sequential model.
        """
        stacked_layers = list()
        # Multiple cnn block
        for i in range(self.num_cnn_blocks):
            stacked_layers.extend([
                keras.layers.Conv2D(self.num_filters[i], self.kernal_sz[i], strides=self.strides_sz, data_format=self.data_format, input_shape=(self.img_shape[0], self.img_shape[1], self.img_shape[2])),
                keras.layers.MaxPool2D(pool_size=self.pool_sz[i])
            ])
        # Add model tail
        stacked_layers.extend([
            keras.layers.Flatten(),
            keras.layers.Dense(units=self.dense_layer_size, activation=self.activation),
            keras.layers.Dense(units=self.dense_layer_size, activation=self.activation),
            keras.layers.Dense(units=self.num_classes, activation=self.output_activation)
        ])
        model_ins = keras.Sequential(stacked_layers)
        return model_ins