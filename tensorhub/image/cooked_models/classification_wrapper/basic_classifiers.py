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
    def __init__(self, img_shape, num_classes, num_channels=3, num_cnn_blocks=2, conv_sz=None, filter_sz=None, pool_sz=2):
        self.img_shape = img_shape
        self.num_cnn_blocks = num_cnn_blocks
        self.num_channels = num_channels
        self.pool_sz = pool_sz if pool_sz != None else [2]*(self.num_cnn_blocks)
        self.filter_sz = filter_sz if filter_sz != None else [32]*(self.num_cnn_blocks)
        self.conv_sz = conv_sz if conv_sz != None else [2]*(self.num_cnn_blocks)
        self.num_classes = num_classes
        if self.num_classes == 1:
            self.output_activation = "sigmoid"
        else:
            self.output_activation = "softmax"

    @property
    def model(self):
        stacked_layers = list()
        # Multiple CNN blocks
        for i in range(self.num_cnn_blocks):
            stacked_layers.extend([
                keras.layers.Conv2D(self.filter_sz[i], self.conv_sz[i], border_mode="same", input_shape=(self.img_shape[0], self.img_shape[1], self.num_channels)),
                keras.layers.MaxPool2D(pool_size=self.pool_sz)
            ])
        # Adding dense layers
        stacked_layers.extend([
            keras.layers.Flatten(),
            keras.layers.Dense(units=2048, activation="relu"),
            keras.layers.Dense(units=2048, activation="relu"),
            keras.layers.Dense(units=self.num_classes, activation=self.output_activation)
        ])
        model = keras.Sequential(stacked_layers)
        return model