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


def relu(x, alpha=0., max_value=None, threshold=0.):
    """Rectified Linear Unit.
    With default values, it returns element-wise `max(x, 0)`.
    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.

    Arguments:
        x {tensor} -- Input float tensor to perform activation.
        alpha {float} -- Slope of the negative part. Defaults to zero.
        max_value {float} -- Saturation threshold.
        threshold {float} -- Threshold value for thresholded activation.
    Returns:
        tensor -- Output of RELU activation.
    """
    if max_value is None:
        max_value = np.inf
    above_threshold = x * (x >= threshold)
    above_threshold = np.clip(above_threshold, 0.0, max_value)
    below_threshold = alpha * (x - threshold) * (x < threshold)
    return below_threshold + above_threshold

def gelu(x):
    """Gaussian Error Linear Unit. This is a smoother version of the RELU.

    Arguments:
        x {tensor} -- Input float tensor to perform activation.
    
    Returns:
        tensor -- Output of GELU activation.
    """
    return x * 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.pow(x, 3)))))

def linear(x):
    """Linear activation function.

    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Output of linear activation.
    """
    return x

def exponential(x):
    """Exponential activation function.

    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Output of exponential activation.
    """
    return np.exp(x)

def tanh(x):
    """Hyperbolic Tangent (tanh) activation function.

    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Output of tanh activation.
    """
    return np.sinh(x) / np.cosh(x)

def sigmoid(x):
    """Sigmoid activation function. For small values
    (<-5) the sigmoid returns a value close to zero and for larger values (>5)
    the result of the function gets close to 1.
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Output of sigmoid activation.
    """
    return 1.0 / (1.0 + np.exp(-x))

def hard_sigmoid(x):
    """Hard-sigmoid activation function. For small values
    (<-2.5) the sigmoid returns a value zero and for larger values (>+2.5)
    the result of the function gets to 1. For values in between it returns a value `0.2 * x + 0.5`.
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.

    Returns:
        tensor -- Output of sigmoid activation.
    """
    y = 0.2 * x + 0.5
    return np.clip(y, 0, 1)

def softsign(x):
    """Softsign activation function.
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.
    
    Returns:
        tensor -- Output of softsign activation.
    """
    return x / (np.abs(x) + 1)

def softplus(x):
    """Softplus activation function.
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.
    
    Returns:
        tensor -- Output of softplus activation.
    """
    return np.log(np.exp(x) + 1)

def softmax(x, axis=-1):
    """Softmax activation function.

    Arguments:
        x {tensor} -- Input float tensor to perform activation.
        axis {int} -- Integer, axis along which the softmax normalization is applied.
    
    Returns:
        tensor -- Output of softmax transformation.
    
    Raises:
        ValueError: In case `dim(x) == 1`.
    """
    # Dimension of the input tensor
    ndim = x.ndim
    if ndim >= 2:
        y = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return y / np.sum(y, axis=axis, keepdims=True)
    else:
        raise ValueError("Cannot apply softmax to a tensor that is 1D. Received input shape: {}".format(x.shape))

def elu(x, alpha=1.):
    """Exponential linear unit.
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.
        alpha {float} -- A scalar, slope of negative section.
    
    Returns:
        tensor -- Output of exponential linear activation
    """
    return x * (x > 0) + alpha * (np.exp(x) - 1.) * (x < 0)

def selu(x):
    """Scaled Exponential Linear Unit (SELU).

    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are predefined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is "large enough" (see references for more information).
    
    Arguments:
        x {tensor} -- Input float tensor to perform activation.
    
    Returns:
       tensor -- The scaled exponential unit activation: `scale * elu(x, alpha)`.
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)