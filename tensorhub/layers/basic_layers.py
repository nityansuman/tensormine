""" 
@Author: Kumar Nityan Suman
@Date: 2019-06-27 02:34:31

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
import tensorflow as tf
from tensorflow import keras


class Linear(keras.layers.Layer):
    """Standard `Linear transfromation` layer implementation."""

    def __init__(self, units=128):
        """Initialize variables.
        
        Keyword Arguments:
            units {int} -- Number of nodes in the layer. (default: {128})
        """
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        """The __call__ method of your layer will automatically run build the first time it is called.
        You now have a layer that's lazy and easy to use.
        
        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        """Forward pass over the constructed linear transformation layer.
        
        Arguments:
            inputs {tensor} -- Input tensor to the layer.
        
        Returns:
            tensor -- Linearly transformed input tensor.
        """
        return tf.matmul(inputs, self.w) + self.b