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


class AdditiveAttention(tf.keras.layers.Layer):
	"""Additive attention layer, a.k.a. Bahdanau-style attention.

	Arguments:
		tf {cls} -- Base module class.
	"""
	def __init__(self, num_output, use_scale=True, name="bahdanau_attention", **kwargs):
		super(AdditiveAttention, self).__init__(name=name, **kwargs)
		self.num_outputs = num_output
		self.use_scale = use_scale

	def build(self, input_shape):
		"""Initialize input dependent variables.
		
		Arguments:
			input_shape {tensor} -- Input tensor shape.
		"""
		self.W1 = self.add_variable("w1", shape=(int(input_shape[-1]), self.num_outputs))
		self.W2 = self.add_variable("w2", shape=(int(input_shape[-1]), self.num_outputs))
		self.V = self.add_variable("value", shape=(1, int(input_shape[-1])))

	def call(self, query, value):
		"""Forward pass over the layer.

		Arguments:
			query {Tensor} -- Query `Tensor` of shape `[batch_size, Tq, dim]`.
			value {Tensor} -- Query `Tensor` of shape `[batch_size, Tq, dim]`.

		Returns:
			[Tensor] -- Attention output of shape `[batch_size, Tq, dim]`.
		"""
		hidden_state = tf.expand_dims(query, 1)
		score = self.V(keras.activations.tanh(tf.matmul(self.W1, value) + tf.matmul(self.W2, hidden_state)))
		attention_weights = tf.nn.softmax(score, axis=1)
		context_vector = tf.reduce_sum(attention_weights * value, axis=1)
		return context_vector