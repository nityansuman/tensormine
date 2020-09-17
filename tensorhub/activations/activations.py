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

import tensorflow as tf
import numpy as np


def relu(x, alpha=0., max_value=None, threshold=0.):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Keyword Arguments:
		alpha {[type]} -- [description] (default: {0.})
		max_value {[type]} -- [description] (default: {None})
		threshold {[type]} -- [description] (default: {0.})

	Returns:
		[type] -- [description]
	"""
	if max_value == None:
		max_value = np.inf
	above_threshold = x * (x >= threshold)
	above_threshold = tf.clip_by_value(above_threshold, 0.0, max_value)
	below_threshold = alpha * (x - threshold) * (x < threshold)
	return below_threshold + above_threshold


def gelu(x):
	"""Gaussian Error Linear Unit. This is a smoother version of the RELU.

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	return x * 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))


def linear(x):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	return x


def exponential(x):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	return tf.math.exp(x)


def tanh(x):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	return tf.math.sinh(x) / tf.math.cosh(x)


def sigmoid(x):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	return 1.0 / (1.0 + tf.math.exp(-x))


def hard_sigmoid(x):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	y = 0.2 * x + 0.5
	return tf.clip_by_value(y, 0, 1)


def softsign(x):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	return x / (tf.math.abs(x) + 1)


def softplus(x):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	return tf.math.log(tf.math.exp(x) + 1)


def softmax(x, axis=-1):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Keyword Arguments:
		axis {int} -- [description] (default: {-1})

	Raises:
		ValueError: [description]

	Returns:
		[type] -- [description]
	"""
	ndim = x.ndim
	if ndim >= 2:
		y = tf.math.exp(x - np.max(x, axis=axis, keepdims=True))
		return y / np.sum(y, axis=axis, keepdims=True)
	else:
		raise ValueError("Cannot apply softmax to a tensor that is 1D. Received input shape: {}".format(x.shape))


def elu(x, alpha=1.):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Keyword Arguments:
		alpha {[type]} -- [description] (default: {1.})

	Returns:
		[type] -- [description]
	"""
	return x * (x > 0) + alpha * (tf.math.exp(x) - 1.) * (x < 0)


def selu(x):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	alpha = 1.6732632423543772848170429916717
	scale = 1.0507009873554804934193349852946
	return scale * elu(x, alpha)

def mish(x):
	"""[summary]

	Arguments:
		x {[type]} -- [description]

	Returns:
		[type] -- [description]
	"""
	return x * tanh(softplus(x))
