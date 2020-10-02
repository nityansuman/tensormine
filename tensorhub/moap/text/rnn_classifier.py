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


class RNNClassifier(tf.keras.Model):
	"""RNN Model class - A simple recurrent neural network based text classifier.

	Args:
		tf (cls): Parent `Model` class.
	"""
	def __init__(self, vocab_size, num_classes, max_seq_len=256, embedding_size=300, ltype="lstm"):
		"""Model constructor.

		Args:
			vocab_size (int): Size of the token (word/character) vocabulary.
			num_classes (int): Number of target classes.
			max_seq_len (int, optional): Length of the maximum sequence input. Defaults to 256.
			embedding_size (int, optional): Size of embedding to be learned. Defaults to 300.
			ltype (str, optional): Type of RNN network to build (Options: "lstm", "gru", "vanilla"). Defaults to "lstm".
		"""
		super(RNNClassifier, self).__init__()
		self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_seq_len)
		if ltype == "lstm":
			self.layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=max_seq_len, return_sequences=True))
			self.layer_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=max_seq_len))
		elif ltype == "gru":
			self.layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=max_seq_len, return_sequences=True))
			self.layer_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=max_seq_len))
		else:
			self.layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=max_seq_len, return_sequences=True))
			self.layer_2 = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=max_seq_len))
		self.d1 = tf.keras.layers.Dense(units=max_seq_len*2)
		self.d2 = tf.keras.layers.Dense(units=max_seq_len*2)
		self.output = tf.keras.layers.Dense(units=num_classes, activation="softmax")
		self.batch_normalization = tf.keras.layers.BatchNormalization()
		self.dropout = tf.keras.layers.Dropout(rate=0.3)
	
	def call(self, x):
		"""Forward pass of the model.

		Args:
			x (tensor): Input tensor.

		Returns:
			tensor: Output tensor.
		"""
		x = self.embedding(x)
		x = self.batch_normalization(x)
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = self.dropout(x)
		x = self.d1(x)
		x = self.d2(x)
		x = self.dropout(x)
		x = self.output(x)
		return x