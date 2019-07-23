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
from tensorhub.layers.bert.config import BertConfig
from tensorhub.layers.bert.embeddings import embedding_lookup, embedding_postprocessor
from tensorhub.layers.bert.transformer import transformer_model
from tensorhub.layers.bert.utils import *


class BertLayer(Layer):
  """BERT model ("Bidirectional Embedding Representations from a Transformer").
  Example usage:
  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
  model = modeling.BertModel(config=config, is_training=True)
  label_embeddings = tf.Variable(...)
  pooled_output = model(input_ids, input_mask, token_type_ids) # return pooled_output by default
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """
  def __init__(self, config, is_training, name="bert"):
    """Constructor for BertModel.
    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      name: (optional) Keras model name. Defaults to "bert".
    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
  
    super(BertLayer, self).__init__(name=name)
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0
    
    self.embedding_lookup = embedding_lookup(name="embeddings",
                                             vocab_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             initializer_range=config.initializer_range,
                                             word_embedding_name="word_embeddings")
    self.embedding_postprocessor = embedding_postprocessor(name="embeddings",
                                                           use_token_type=True,
                                                           token_type_vocab_size=config.type_vocab_size,
                                                           token_type_embedding_name="token_type_embeddings",
                                                           use_position_embeddings=True,
                                                           position_embedding_name="position_embeddings",
                                                           initializer_range=config.initializer_range,
                                                           max_position_embeddings=config.max_position_embeddings,
                                                           dropout_prob=config.hidden_dropout_prob)
    self.transformer_model = transformer_model(name="encoder",
                                               hidden_size=config.hidden_size,
                                               num_hidden_layers=config.num_hidden_layers,
                                               num_attention_heads=config.num_attention_heads,
                                               intermediate_size=config.intermediate_size,
                                               intermediate_act_fn=get_activation(config.hidden_act),
                                               hidden_dropout_prob=config.hidden_dropout_prob,
                                               attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                               initializer_range=config.initializer_range)
    self.pooler = tf.keras.layers.Dense(name="pooler/dense",
                                                units=config.hidden_size,
                                                activation=tf.tanh,
                                                kernel_initializer=create_initializer(config.initializer_range))
  
  def call(self,
           input_ids,
           input_mask=None,
           token_type_ids=None, sequence_output=False):
    """Caller of BertModel.
    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
    
    Returns:
      All encoder outputs.
    """

    input_shape = get_shape_list(input_ids, expected_rank=2)
    # print("input shape", input_shape)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    # Perform embedding lookup on the word ids.
    self.embedding_output = self.embedding_lookup(inputs=input_ids)

    # Add positional embeddings and token type embeddings, then layer
    # normalize and perform dropout.
    self.embedding_output = self.embedding_postprocessor(inputs=self.embedding_output,
                                                         token_type_ids=token_type_ids)

    # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
    # mask of shape [batch_size, seq_length, seq_length] which is used
    # for the attention scores.
    attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

    # Run the stacked transformer.
    # `sequence_output` shape = [batch_size, seq_length, hidden_size].
    self.all_encoder_layers = self.transformer_model(inputs=self.embedding_output,
                                                     attention_mask=attention_mask,
                                                     do_return_all_layers=True)

    self.sequence_output = self.all_encoder_layers[-1]
    # The "pooler" converts the encoded sequence tensor of shape
    # [batch_size, seq_length, hidden_size] to a tensor of shape
    # [batch_size, hidden_size]. This is necessary for segment-level
    # (or segment-pair-level) classification tasks where we need a fixed
    # dimensional representation of the segment.

    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. We assume that this has been pre-trained
    first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
    self.pooled_output = self.pooler(first_token_tensor)
    # x = tf.keras.layers.Dense(20)(self.pooled_output)

    if sequence_output:
    	return self.get_embedding_output()
    else:
    	return self.get_pooled_output()

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.
    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).
    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_lookup.embedding_table

  def map_to_stock_variable_name(self, name, prefix="bert"):
    name = name.split(":")[0].replace("embeddings_1", "embeddings")
    return name

  def load_stock_weights(self, bert, ckpt_file):
    ckpt_reader = tf.train.load_checkpoint(ckpt_file)

    bert_prefix = bert.weights[0].name.split("/")[0]
    # print(bert_prefix)
    weights = []
    for idx, weight in enumerate(bert.weights):
        # print(idx, weight.name)
        stock_name = self.map_to_stock_variable_name(weight.name, bert_prefix)
        if ckpt_reader.has_tensor(stock_name):
            value = ckpt_reader.get_tensor(stock_name)
#             print("stock_name" , stock_name, "value", value)
            weights.append(value)
        else:
#             print("stock_name" , stock_name)
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(weight.name, stock_name, ckpt_file))
            # raise ValueError("No value for:[{}], i.e.:[{}] in:[{}]".format(weight.name, stock_name, ckpt_file))
            weights.append(weight.value())
    bert.set_weights(weights)
    print("Done loading {} BERT weights from: {} into {} (prefix:{})".format(
        len(weights), ckpt_file, bert, bert_prefix))



