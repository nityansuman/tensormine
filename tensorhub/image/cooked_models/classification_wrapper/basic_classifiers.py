"""
@Author: Kumar Nityan Suman
@Date: 2019-06-27 03:33:41

Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Load packages
from tensorflow import keras


class CNNClassifier:
    def __init__(self, img_shape, num_channels=3, num_classes, num_cnn_blocks=2, conv_sz=None, filter_sz=None, pool_sz=2):
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