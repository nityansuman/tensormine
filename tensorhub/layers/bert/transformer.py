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

import re
import six
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorhub.layers.bert.attention import attention_layer
from tensorhub.layers.bert.utils import *


class transformer_model(Model):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".
  This is almost an exact implementation of the original Transformer encoder.
  See the original paper:
  https://arxiv.org/abs/1706.03762
  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  """
  def __init__(self, 
               name,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_act_fn=gelu,
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               initializer_range=0.02):
    """Constructor of transformer_model
    Args:
      name: model name
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
    """
    super(transformer_model, self).__init__(name=name)
    
    if hidden_size % num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, num_attention_heads))
    attention_head_size = int(hidden_size / num_attention_heads)
    
    self.attention_heads = []
    self.attention_outputs = []
    self.attention_layer_norms = []
    self.intermediate_outputs = []
    self.layer_outputs = []
    self.output_layer_norms = []
    for layer_idx in range(num_hidden_layers):
      attention_head = attention_layer(
                name="layer_%d" % layer_idx + "/attention" + "/self", 
                num_attention_heads=num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range)
      self.attention_heads.append(attention_head)
    
      attention_output = tf.keras.layers.Dense(                
                hidden_size,
                name="layer_%d" % layer_idx + "/attention" + "/output" + "/dense",
                kernel_initializer=create_initializer(initializer_range))
      self.attention_outputs.append(attention_output)
    
      attention_layer_norm = layer_norm(name="layer_%d" % layer_idx + "/attention/output/LayerNorm")
      self.attention_layer_norms.append(attention_layer_norm)

      intermediate_output = tf.keras.layers.Dense(
              intermediate_size,
              name="layer_%d" % layer_idx + "/intermediate" + "/dense",
              activation=intermediate_act_fn,
              kernel_initializer=create_initializer(initializer_range))
      self.intermediate_outputs.append(intermediate_output)

      layer_output = tf.keras.layers.Dense(
              hidden_size,
              name="layer_%d" % layer_idx + "/output" + "/dense",
              kernel_initializer=create_initializer(initializer_range))
      self.layer_outputs.append(layer_output)
      
      output_layer_norm = layer_norm(name="layer_%d" % layer_idx + "/output/LayerNorm")
      self.output_layer_norms.append(output_layer_norm)
    
    self.dropout = tf.keras.layers.Dropout(rate=hidden_dropout_prob)
    
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
  
  def call(self,
           inputs,
           attention_mask=None,
           do_return_all_layers=False):
    '''
    Args:
      inputs: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      
      do_return_all_layers: Whether to also return all layers or just the final
        layer.
    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.
    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    '''
    input_shape = get_shape_list(inputs, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]
    
    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != self.hidden_size:
      raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                       (input_width, self.hidden_size))
    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(inputs)
    
    all_layer_outputs = []
    for layer_idx in range(self.num_hidden_layers):
      layer_input = prev_output
      
      attention_heads = []
      attention_head = self.attention_heads[layer_idx](inputs=None,
                                                       from_tensor=layer_input,
                                                       to_tensor=layer_input,
                                                       attention_mask=attention_mask,
                                                       do_return_2d_tensor=True,
                                                       batch_size=batch_size,
                                                       from_seq_length=seq_length,
                                                       to_seq_length=seq_length)
      attention_heads.append(attention_head)
          
      attention_output = None
      if len(attention_heads) == 1:
        attention_output = attention_heads[0]
      else:
        # In the case where we have other sequences, we just concatenate
        # them to the self-attention head before the projection.
        attention_output = tf.concat(attention_heads, axis=-1)
      
      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      attention_output = self.attention_outputs[layer_idx](attention_output)
      attention_output = self.dropout(attention_output)
      attention_output = self.attention_layer_norms[layer_idx](attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      intermediate_output = self.intermediate_outputs[layer_idx](attention_output)

      # Down-project back to `hidden_size` then add the residual.
      layer_output = self.layer_outputs[layer_idx](intermediate_output)
      layer_output = self.dropout(layer_output)
      layer_output = self.output_layer_norms[layer_idx](layer_output + attention_output)
      prev_output = layer_output
      all_layer_outputs.append(layer_output)
    
    if do_return_all_layers:
      final_outputs = []
      for layer_output in all_layer_outputs:
        final_output = reshape_from_matrix(layer_output, input_shape)
        final_outputs.append(final_output)
      return final_outputs
    else:
      final_output = reshape_from_matrix(prev_output, input_shape)
      return final_output
