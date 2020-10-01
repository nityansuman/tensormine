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


class ConvNet(tf.keras.Model):
	"""ConvNet Model class - A simple CNN based model developed on MNIST dataset for image classification.

	Args:
		tf (cls): Inhereting parent `Model` class.
	"""
	def __init__(self, num_classes, name=None):
		"""Model constructor.

		Args:
			num_classes (int): Number of target classes.
			name (str, optional): Name of the model. Defaults to None.
		"""
		super(ConvNet, self).__init__(name=name)
		self.conv_2d_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
		self.conv_2d_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")
		self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
		self.flatten = tf.keras.layers.Flatten()
		self.dropout = tf.keras.layers.Dropout(rate=0.5)
		self.dense = tf.keras.layers.Dense(units=num_classes, activation="softmax")
	
	def call(self, x):
		"""Foreward pass of the model.

		Args:
			x (tensor): Input tensor.

		Returns:
			tensor: Output tensor.
		"""
		x = self.conv_2d_1(x)
		x = self.max_pool(x)
		x = self.conv_2d_2(x)
		x = self.max_pool(x)
		x = self.flatten(x)
		x = self.dropout(x)
		x = self.dense(x)
		return x
