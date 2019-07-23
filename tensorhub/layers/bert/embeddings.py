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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorhub.layers.bert.utils import *


class embedding_lookup(Layer):
  """Looks up words embeddings for id tensor."""
  def __init__(self,
               name,
               vocab_size,
               embedding_size=128,
               initializer_range=0.02,
               word_embedding_name="word_embeddings"):
    '''
    Constructor for embedding_lookup.
    
    Args:
      name: layer name.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
    '''
    super(embedding_lookup, self).__init__(name=name)    
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.initializer_range = initializer_range
    self.word_embedding_name = word_embedding_name

  def build(self, input_shape):
    self.embedding_table = self.add_variable(name=self.word_embedding_name,
                                             shape=[self.vocab_size, self.embedding_size],
                                             initializer=create_initializer(self.initializer_range),
                                             dtype=tf.float32)  
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  def call(self, inputs):
    '''
    Args:
      inputs: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    '''
  
    if inputs.shape.ndims == 2:
      inputs = tf.expand_dims(inputs, axis=[-1])
    output = tf.nn.embedding_lookup(self.embedding_table, inputs)
    input_shape = get_shape_list(inputs)
    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
    
    return output

class embedding_postprocessor(Layer):
  def __init__(self,
               name,
               use_token_type=False,
               token_type_vocab_size=16,
               token_type_embedding_name="token_type_embeddings",
               use_position_embeddings=True,
               position_embedding_name="position_embeddings",
               initializer_range=0.02,
               max_position_embeddings=512,
               dropout_prob=0.1):
    """
    Constructor for embedding_postprocessor.
    
    Args:
      name: layer name.
      use_token_type: bool. Whether to add embeddings for `token_type_ids`.
      token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
      token_type_embedding_name: string. The name of the embedding table variable
        for token type ids.
      use_position_embeddings: bool. Whether to add position embeddings for the
        position of each token in the sequence.
      position_embedding_name: string. The name of the embedding table variable
        for positional embeddings.
      initializer_range: float. Range of the weight initialization.
      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      dropout_prob: float. Dropout probability applied to the final output tensor.
    """

    super(embedding_postprocessor, self).__init__(name=name)
    self.use_token_type = use_token_type
    self.token_type_vocab_size = token_type_vocab_size
    self.token_type_embedding_name = token_type_embedding_name
    self.use_position_embeddings = use_position_embeddings
    self.position_embedding_name = position_embedding_name
    self.initializer_range = initializer_range
    self.max_position_embeddings = max_position_embeddings
    
    self.dropout = tf.keras.layers.Dropout(rate=dropout_prob)
    self.layer_norm = layer_norm(name="LayerNorm")
    
  def build(self, input_shape):
    width = input_shape[2]
    if self.use_token_type:
      self.token_type_table = self.add_variable(name=self.token_type_embedding_name,
                                                shape=[self.token_type_vocab_size, width],
                                                initializer=create_initializer(self.initializer_range))
    
    if self.use_position_embeddings:
      self.full_position_embeddings = self.add_variable(name=self.position_embedding_name,
                                                        shape=[self.max_position_embeddings, width],
                                                        initializer=create_initializer(self.initializer_range))

  def call(self, inputs, token_type_ids):
    """Performs various post-processing on a word embedding tensor.
    
    Args:
      inputs: float Tensor of shape [batch_size, seq_length,
        embedding_size].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
    Returns:
      float tensor with same shape as `input`.
    Raises:
      ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(inputs, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]
    
    if seq_length > self.max_position_embeddings:
      raise ValueError("The seq length (%d) cannot be greater than "
                       "`max_position_embeddings` (%d)" %
                       (seq_length, self.max_position_embeddings))
    output = inputs
    
    if self.use_token_type:
      if token_type_ids is None:
        raise ValueError("`token_type_ids` must be specified if"
                         "`use_token_type` is True.")
      # This vocab will be small so we always do one-hot here, since it is always
      # faster for a small vocabulary.
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
      token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
      output += token_type_embeddings
    
    if self.use_position_embeddings:
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      if seq_length < self.max_position_embeddings:
        position_embeddings = tf.slice(self.full_position_embeddings, [0, 0],
                                       [seq_length, -1])
      else:
        position_embeddings = self.full_position_embeddings

      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings
    
    output = self.layer_norm(output)
    output = self.dropout(output)
    return output
