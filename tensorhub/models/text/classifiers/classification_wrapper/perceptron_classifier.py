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

# Load packages
from tensorflow import keras
from tensorhub.utilities.activations import relu, sigmoid, softmax


class PerceptronClassifier(keras.models.Model):
    """Text sequence classification with Multi-layer perceptron."""

    def __init__(self, vocab_size, num_classes, max_seq_length=256, dp_rate=0.4, act=relu, output_act=softmax, embedding_dim=100, learn_embedding=True, embedding_matrix=None):
        """Class constructor to initialize member variables.
        
        Arguments:
            vocab_size {int} -- Number of tokens in the vocabulary.
            num_classes {int} -- Number of prediction classes.
        
        Keyword Arguments:
            max_seq_length {int} -- Max. length of an input sequence. (default: {256})
            dp_rate {float} -- Float value for `dropout rates`. If no dropout keep it 0. (default: {0.4})
            act {str} -- Activation to be used for dense layers. (default: {relu})
            output_act {str} -- Activation to be used witth output activation. (default: {softmax})
            embedding_dim {int} -- Size of the embedding to be learned or otherwise. (default: {100})
            learn_embedding {bool} -- Set boolean flag to `True` to learn embedding as part of the neural network. (default: {True})
            embedding_matrix {numpy-array} -- if `learn_embedding` is `False`, use this to load pre-trained embedding vectors. (default: {None})
        """
        super(PerceptronClassifier, self).__init__()
        # Set member variables
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.act = activation
        self.output_act = output_activation
        self.max_seq_length = max_seq_length
        self.dp_rate = dp_rate
        self.learn_embedding = learn_embedding
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        # Define layers
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_seq_length)
        else:
            self.embedding_layer = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_seq_length, trainable=False, weights=[self.embedding_matrix])
        self.dropout_layer = keras.layers.Dropout(rate=self.dp_rate)
        self.d1 = keras.layers.Dense(units=self.max_seq_length, activation=self.act)
        self.d2 = keras.layers.Dense(units=self.max_seq_length / 4, activation=self.act)
        self.d3 = keras.layers.Dense(units=self.max_seq_length / 2, activation=self.act)
        self.d4 = keras.layers.Dense(units=self.num_classes, activation=self.output_act)

    def call(self, x):
        """Forward pass over the network.
        
        Returns:
            output -- Output tensor from the network.
        """
        x = self.embedding_layer(x)
        x = self.dropout_layer(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.dropout_layer(x)
        x = self.d3(x)
        x = self.d4(x)
        return x