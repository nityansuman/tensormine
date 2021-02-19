# Copyright 2021 The TensorHub Authors. All Rights Reserved.
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

def softplus(x):
	"""Compute SOFTPLUS activation over tensor.

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return tf.math.log(tf.math.exp(x) + 1)

def mish(x):
	"""Compute MISH activation over the input tensor. A Self Regularized Non-Monotonic Neural Activation Function.

	Args:
		x (tensor): Input tensor.

	Returns:
		tensor: Activated input tensor. Tensor will be of the same shape and dtype of input `x`.
	"""
	return x * tanh(softplus(x))
