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


class DimReductionLayer(tf.keras.layers.Layer):
	"""Dimensionality reduction `Inception v1` module implemented as a layer for advance feature creation.
	
	Know more at: https://arxiv.org/pdf/1409.4842v1.pdf
	"""
	def __init__(self, num_filters=64, name=None):
		"""Initialize variables.

		Keyword Arguments:
			num_filters {int} -- Number of filters. (default: {64})
			name {str} -- Name of the layer. (default: {None})
		"""
		super(DimReductionLayer, self).__init__(name=name)
		self.conv2d_1a = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1))
		self.conv2d_1b = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1))
		self.conv2d_1c = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1))
		self.conv2d_1d = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1))
		self.conv2d_3 = tf.keras.layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=(1, 1))
		self.conv2d_5 = tf.keras.layers.Conv2D(filters=num_filters/2, kernel_size=(5, 5), strides=(1, 1))
		self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1))
		self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
	
	def call(self, x):
		"""Forward pass of the layer.

		Args:
			x (tensor): Input tensor.

		Returns:
			tensor: Output tensor.
		"""
		x1 = self.conv2d_1a(x)
		
		x2 = self.conv2d_1b(x)
		x2 = self.conv2d_3(x2)
		
		x3 = self.conv2d_1c(x)
		x3 = self.conv2d_5(x3)
		
		x4 = self.max_pool(x)
		x4 = self.conv2d_1d(x4)
		output = self.concat_layer([x1, x2, x3, x4])
		return output


class NaiveLayer(tf.keras.layers.Layer):
	"""Naive `Inception v1` module implemented as a layer for advance feature creation.
	
	Know more at: https://arxiv.org/pdf/1409.4842v1.pdf
	"""
	def __init__(self, num_filters=64, name=None):
		"""Initialize variables.

		Keyword Arguments:
			num_filters {int} -- Number of filters. (default: {64})
			name {str} -- Name of the layer. (default: {None})
		"""
		super(NaiveLayer, self).__init__(name=name)
		self.conv2d_1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1))
		self.conv2d_3 = tf.keras.layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=(1, 1))
		self.conv2d_5 = tf.keras.layers.Conv2D(filters=num_filters/2, kernel_size=(5, 5), strides=(1, 1))
		self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1))
		self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
	
	def call(self, x):
		"""Forward pass of the layer.

		Args:
			x (tensor): Input tensor.

		Returns:
			tensor: Output tensor.
		"""
		x1 = self.conv2d_1(x)
		x2 = self.conv2d_3(x)
		x3 = self.conv2d_5(x)
		x4 = self.max_pool(x)
		output = self.concat_layer([x1, x2, x3, x4])
		return output