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
from tensorflow.keras import layers


class ConvNet(tf.keras.Model):
	"""ConvNet image classifier.

	Args:
		tf (cls):  Parent `Model` class.
	"""
	def __init__(self, num_classes, name="ConvNet"):
		"""Model constructor.

		Args:
			num_classes (int): Number of target classes.
			name (str, optional): Name of the model. Defaults to None.
		"""
		super(ConvNet, self).__init__(name=name)
		self.conv_2d_1 = layers.Conv2D(filters=32, kernel_size=3, activation="relu")
		self.conv_2d_2 = layers.Conv2D(filters=64, kernel_size=3, activation="relu")
		self.max_pool = layers.MaxPool2D()
		self.flatten = layers.Flatten()
		self.drop_out = layers.Dropout(0.5)
		if num_classes == 1:
			self.dense_1 = layers.Dense(units=num_classes, activation="sigmoid")
		elif num_classes >= 2:
			self.dense_1 = layers.Dense(units=num_classes, activation="softmax")
		else:
			raise ValueError("`num_classes` cannot be Null or negative.")
	
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
		x = self.drop_out(x)
		return self.dense_1(x)
