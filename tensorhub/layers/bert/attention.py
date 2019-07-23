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

class attention_layer(Layer):
  '''Performs multi-headed attention from `from_tensor` to `to_tensor`.
  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.
  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].
  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.
  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.
  '''
  def __init__(self, 
               name, 
               num_attention_heads=1,
               size_per_head=512,
               query_act=None,
               key_act=None,
               value_act=None,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02):
    """Constructor of attention_layer
    Args:
      name: layer name
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
            attention probabilities.
        initializer_range: float. Range of the weight initializer.
    """

    super(attention_layer, self).__init__(name=name)
    
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    
    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_layer` = [B*F, N*H]
    self.query_layer = tf.keras.layers.Dense(num_attention_heads * size_per_head,
                                             activation=query_act,
                                             name="query",
                                             kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    self.key_layer = tf.keras.layers.Dense(num_attention_heads * size_per_head,
                                           activation=key_act,
                                           name="key",
                                           kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    self.value_layer = tf.keras.layers.Dense(num_attention_heads * size_per_head,
                                             activation=value_act,
                                             name="value",
                                             kernel_initializer=create_initializer(initializer_range))
    self.dropout = tf.keras.layers.Dropout(rate=attention_probs_dropout_prob)

  def transpose_for_scores(self, input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor  
  
  def call(self,
           inputs,
           from_tensor,
           to_tensor,
           attention_mask=None,
           do_return_2d_tensor=False,
           batch_size=None,
           from_seq_length=None,
           to_seq_length=None):
    """
    Args:
        inputs: default argument in 'call' function of keras layer.
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
            from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
            from_seq_length, to_seq_length]. The values should be 1 or 0. The
            attention scores will effectively be set to -infinity for any positions in
            the mask that are 0, and will be unchanged for positions that are 1.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
            * from_seq_length, num_attention_heads * size_per_head]. If False, the
            output will be of shape [batch_size, from_seq_length, num_attention_heads
            * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
            of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `to_tensor`.
    Returns:
        float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).
    Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
    
    if len(from_shape) != len(to_shape):
      raise ValueError(
          "The rank of `from_tensor` must match the rank of `to_tensor`.")
    
    if len(from_shape) == 3:
      batch_size = from_shape[0]
      from_seq_length = from_shape[1]
      to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
      if (batch_size is None or from_seq_length is None or to_seq_length is None):
        raise ValueError(
            "When passing in rank 2 tensors to attention_layer, the values "
            "for `batch_size`, `from_seq_length`, and `to_seq_length` "
            "must all be specified.")

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)
    
    query_layer = self.query_layer(from_tensor_2d)
    key_layer = self.key_layer(to_tensor_2d)
    value_layer = self.value_layer(to_tensor_2d)
    
    # `query_layer` = [B, N, F, H]
    query_layer = self.transpose_for_scores(query_layer, batch_size,
                                            self.num_attention_heads, from_seq_length,
                                            self.size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = self.transpose_for_scores(key_layer, batch_size, self.num_attention_heads,
                                          to_seq_length, self.size_per_head)
    
    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self.size_per_head)))
    
    if attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder
    
    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    
    if do_return_2d_tensor:
      # `context_layer` = [B*F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size * from_seq_length, self.num_attention_heads * self.size_per_head])
    else:
      # `context_layer` = [B, F, N*V]
      context_layer = tf.reshape(
          context_layer,
          [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head])

    return context_layer
