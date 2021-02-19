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
from tensorflow.keras import layers


class TextCNN(tf.keras.Model):
	"""TextCNN classifier model based on 1D CNN.

	Args:
		tf (cls): Base abstract `Model` class.
	"""
	def __init__(self, vocab_size, embedding_dim=100, num_classes=1, name="TextCNN"):
		super(TextCNN, self).__init__(name=name)
		self.embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
		self.drop_out = layers.Dropout(0.5)
		self.conv_1d_1 = layers.Conv1D(filters=128, kernel_size=7, padding="valid", activation="relu", strides=3)
		self.conv_1d_2 = layers.Conv1D(filters=128, kernel_size=7, padding="valid", activation="relu", strides=3)
		self.global_max_pool = layers.GlobalMaxPool1D()
		self.dense_1 = layers.Dense(units=128, activation="relu")
		if num_classes == 1:
			self.dense_2 = layers.Dense(units=num_classes, activation="sigmoid", name="predictions")
		elif num_classes >= 2:
			self.dense_2 = layers.Dense(units=num_classes, activation="softmax", name="predictions")
		else:
			raise ValueError("`num_classes` cannot be Null or negative.")
	
	def call(self, x):
		"""Forward pass of the model.

		Args:
			x (tensor): Input tensor.

		Returns:
			tensor: Output tensor.
		"""
		embedding = self.embedding_layer(x)
		y = self.drop_out(embedding)
		y = self.conv_1d_1(y)
		y = self.conv_1d_2(y)
		y = self.global_max_pool(y)
		y = self.dense_1(y)
		y = self.drop_out(y)
		return self.dense_2(y)
