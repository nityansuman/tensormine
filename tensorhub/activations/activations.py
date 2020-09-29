# Copyright 2020 The TensorHub Authors. All Rights Reserved.
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

# Import packages
import tensorflow as tf
import numpy as np


def relu(x, alpha=0.0, max_value=None, threshold=0):
	"""Compute RELU activate (Rectified Error Linear Unit) activation over input tensor.
	
	With default values, this returns the standard ReLU activation:
	`max(x, 0)`, the element-wise maximum of 0 and the input tensor.

	Args:
		x (tensor): Input tensor.
		alpha (float, optional): Governs the slope for values lower than the threshold. Defaults to 0.0.
		max_value (float, optional): Sets the saturation threshold i.e., largest value the function will return. Defaults to None.
		threshold (int, optional): Threshold value below which values will be damped or set to zero. Defaults to 0.0.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	if max_value == None:
		max_value = np.inf
	above_threshold = x * (x >= threshold)
	above_threshold = tf.clip_by_value(above_threshold, 0.0, max_value)
	below_threshold = alpha * (x - threshold) * (x < threshold)
	return below_threshold + above_threshold


def gelu(x):
	"""Compute GELU (Gaussian Error Linear Unit) activation over input tensor. This is a smoother version of the RELU.
	
	Gaussian error linear unit (GELU) computes `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
	The (GELU) nonlinearity weights inputs by their value, rather than gates inputs by their sign as in ReLU.

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return x * 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))


def linear(x):
	"""Compute LINEAR activation over input tensor.

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return x


def exponential(x):
	"""Compute EXPONENTIAL activation over input tensor.

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return tf.math.exp(x)


def tanh(x):
	"""Compute TANH activation over input tensor.
	Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear.
	But unlike Sigmoid, its output is zero-centered.
	Therefore, in practice the tanh non-linearity is always preferred to the sigmoid nonlinearity.

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return tf.math.sinh(x) / tf.math.cosh(x)


def sigmoid(x):
	"""Compute SIGMOID acttivation over input tensor.
	Sigmoid takes a real value as input and outputs another value between 0 and 1.
	It’s non-linear, continuously differentiable, monotonic, and has a fixed output range.
	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return 1.0 / (1.0 + tf.math.exp(-x))


def softsign(x):
	"""Compute SOFTSIGN activation over tensor.

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return x / (tf.math.abs(x) + 1)


def softplus(x):
	"""Compute SOFTPLUS activation over tensor.

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return tf.math.log(tf.math.exp(x) + 1)


def softmax(x, axis=-1):
	"""Compute SOFTMAX activation over tensor.
	Softmax function calculates the probabilities distribution of the event over ‘n’ different events.
	In general way of saying, this function will calculate the probabilities of each target class over all possible target classes. 

	Args:
		x (tensor): Input tensor.
		axis (int): Tensor axis index.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	ndim = x.ndim
	if ndim >= 2:
		y = tf.math.exp(x - np.max(x, axis=axis, keepdims=True))
		return y / np.sum(y, axis=axis, keepdims=True)
	else:
		raise ValueError("Cannot apply softmax to a tensor that is 1D. Received input shape: {}".format(x.shape))


def elu(x, alpha=1.0):
	"""Compute ELU (Exponential Linear Unit) activation over tensor.
	Exponential Linear Unit or its widely known name ELU is a function that tend to converge cost to zero faster and produce more accurate results.
	Different to other activation functions, ELU has a extra alpha constant which should be positive number.
	ELU is very similiar to RELU except negative inputs. They are both in identity function form for non-negative inputs.
	On the other hand, ELU becomes smooth slowly until its output equal to -α whereas RELU sharply smoothes.

	Args:
		x (tensor): Input tensor.
		alpha (float): Floating point smoothing factor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return x * (x > 0) + alpha * (tf.math.exp(x) - 1.0) * (x < 0)


def selu(x):
	"""Compute SELU (Scaled ELU) activation over tensor.
	When using this activation function in practice, one must use lecun_normal for weight initialization, and if dropout wants to be applied, one should use AlphaDropout. 
	The authors have calculated two values; an alpha 
	α and lambda λ value for the equation, which I'm going to show first
		a ≈ 1.6732632423543772848170429916717
		λ ≈ 1.0507009873554804934193349852946

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	alpha = 1.6732632423543772848170429916717
	scale = 1.0507009873554804934193349852946
	return scale * elu(x, alpha)


def mish(x):
	"""Compute MISH activation over the input tensor. A Self Regularized Non-Monotonic Neural Activation Function.

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return x * tanh(softplus(x))
