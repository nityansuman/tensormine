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
import tensorflow as tf


def gelu(x):
    """Gaussian Error Linear Unit. This is a smoother version of the RELU.

    Arguments:
        x {tensor} -- Input float Tensor to perform activation.
    
    Returns:
        tensor -- Input float tensor with the GELU activation applied.
    """
    return x *0.5 * (1.0 + tf.tanh((tf.math.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

def linear(x):
    """Linear activation function.

    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Input float tensor with linear activation applied.
    """
    return x

def exponential(x):
    """Exponential activation function.

    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Input float tensor with exponential activation applied.
    """
    return tf.math.exp(x)

def tanh(x):
    """Hyperbolic Tangent (tanh) activation function.

    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Input float tensor with tanh activation applied.
    """
    return tf.math.sinh(x) / tf.math.cosh(x)

def sigmoid(x):
    """Sigmoid activation function. For small values
    (<-5) the sigmoid returns a value close to zero and for larger values (>5)
    the result of the function gets close to 1.
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Input float tensor with sigmoid activation applied.
    """
    return 1.0 / (1.0 + tf.math.exp(-x))

def hard_sigmoid(x):
    """Hard-sigmoid activation function. For small values
    (<-2.5) the sigmoid returns a value zero and for larger values (>+2.5)
    the result of the function gets to 1. For values in between it returns a value `0.2 * x + 0.5`.
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Input float tensor with sigmoid activation applied.
    """
    return tf.clip_by_value(0.2 * x + 0.5, 0., 1.)

def softsign(x):
    """Softsign activation function.
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.
    
    Returns:
        tensor -- Input float tensor with softsign activation applied.
    """
    return x / (tf.abs(x) + 1)

def softplus(x):
    """Softplus activation function.
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.
    
    Returns:
        tensor -- Input float tensor with softplus activation applied.
    """
    return tf.math.log(tf.math.exp(x) + 1)