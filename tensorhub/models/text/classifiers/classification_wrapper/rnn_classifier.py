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
from tensorhub.utilities.activations import tanh, sigmoid, softmax


class LSTMClassifier(keras.models.Model):
    """Text sequence classification with `Long Short Term Memory`."""

    def __init__(self, vocab_size, num_classes, bidir=False, max_seq_length=256, dense_layer_size=512, dp_rate=0.4, act=tanh, output_act=softmax, embedding_dim=100, learn_embedding=True, embedding_matrix=None):
        """Class constructor to initialize member variables.
        
        Arguments:
            vocab_size {int} -- Number of tokens in the vocabulary.
            num_classes {int} -- Number of prediction classes.
        
        Keyword Arguments:
            bidir {bool} -- Set boolean flag to use bidirectional RNNs. (default: {False})
            max_seq_length {int} -- Max. length of an input sequence. (default: {256})
            dense_layer_size {list} -- Number of nodes in the classification dense layers. (default: {512})
            dp_rate {float} -- Ratio of number of nodes to be droped in the dropout layer. (default: {0.4})
            embedding_dim {int} -- Size of the embedding to be learned or otherwise. (default: {100})
            learn_embedding {bool} -- Set boolean flag to `True` to learn embedding as part of the neural network. (default: {True})
            embedding_matrix {numpy-array} -- if `learn_embedding` is `False`, use this to load pre-trained embedding vectors. (default: {None})
        """
        super(LSTMClassifier, self).__init__()
        # Set member variables
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length        
        self.dense_layer_size = dense_layer_size
        self.dp_rate = dp_rateself.bidir = bidir
        self.act = act
        self.output_act = oputput_act
        self.learn_embedding = learn_embedding
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        # Define layers
        if self.learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_seq_length))
        elif self.learn_embedding == False:
            self.embedding_layer = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[self.embedding_matrix], trainable=False, input_length=self.max_seq_length))
        if self.bidir == False:
            self.lstm1 = keras.layers.LSTM(units=self.max_seq_length, return_sequences=True, activation=self.act)
            self.lstm2 = keras.layers.LSTM(units=self.max_seq_length, activation=self.act)
        else:
            self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(units=self.max_seq_length, return_sequences=True, activation=self.act))
            self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(units=self.max_seq_length, activation=self.act))
        self.d1 = keras.layers.Dense(units=self.dense_layer_size)
        self.d2 = keras.layers.Dense(units=self.dense_layer_size)
        self.d3 = keras.layers.Dense(units=self.num_classes, activation=self.output_act)
        self.norm_layer = keras.layers.BatchNormalization()
        self.dropout_layer = keras.layers.Dropout(rate=self.dp_rate)
    
    def call(self, x):
        """Forward pass over the network.
        
        Returns:
            output -- Output tensor from the network.
        """
        x = self.embedding_layer(x)
        x = self.norm_layer(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dropout_layer(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.dropout_layer(x)
        x = self.d3(x)
        return x

class GRUClassifier(keras.models.Model):
    """Text sequence classification with `Gated Reccurent Units`."""

    def __init__(self, vocab_size, num_classes, bidir=False, max_seq_length=256, dense_layer_size=512, dp_rate=0.4, act=tanh, output_act=softmax, embedding_dim=100, learn_embedding=True, embedding_matrix=None):
        """Class constructor to initialize member variables.
        
        Arguments:
            vocab_size {int} -- Number of tokens in the vocabulary.
            num_classes {int} -- Number of prediction classes.
        
        Keyword Arguments:
            bidir {bool} -- Set boolean flag to use bidirectional RNNs. (default: {False})
            max_seq_length {int} -- Max. length of an input sequence. (default: {256})
            dense_layer_size {list} -- Number of nodes in the classification dense layers. (default: {512})
            dp_rate {float} -- Ratio of number of nodes to be droped in the dropout layer. (default: {0.4})
            embedding_dim {int} -- Size of the embedding to be learned or otherwise. (default: {100})
            learn_embedding {bool} -- Set boolean flag to `True` to learn embedding as part of the neural network. (default: {True})
            embedding_matrix {numpy-array} -- if `learn_embedding` is `False`, use this to load pre-trained embedding vectors. (default: {None})
        """
        super(LSTMClassifier, self).__init__()
        # Set member variables
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length        
        self.dense_layer_size = dense_layer_size
        self.dp_rate = dp_rateself.bidir = bidir
        self.act = act
        self.output_act = oputput_act
        self.learn_embedding = learn_embedding
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        # Define layers
        if self.learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_seq_length))
        elif self.learn_embedding == False:
            self.embedding_layer = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[self.embedding_matrix], trainable=False, input_length=self.max_seq_length))
        if self.bidir == False:
            self.gru1 = keras.layers.GRU(units=self.max_seq_length, return_sequences=True, activation=self.act)
            self.gru2 = keras.layers.GRU(units=self.max_seq_length, activation=self.act)
        else:
            self.gru1 = keras.layers.Bidirectional(keras.layers.GRU(units=self.max_seq_length, return_sequences=True, activation=self.act))
            self.gru2 = keras.layers.Bidirectional(keras.layers.GRU(units=self.max_seq_length, activation=self.act))
        self.d1 = keras.layers.Dense(units=self.dense_layer_size)
        self.d2 = keras.layers.Dense(units=self.dense_layer_size)
        self.d3 = keras.layers.Dense(units=self.num_classes, activation=self.output_act)
        self.norm_layer = keras.layers.BatchNormalization()
        self.dropout_layer = keras.layers.Dropout(rate=self.dp_rate)
    
    def call(self, x):
        """Forward pass over the network.
        
        Returns:
            output -- Output tensor from the network.
        """
        x = self.embedding_layer(x)
        x = self.norm_layer(x)
        x = self.gru1(x)
        x = self.gru2(x)
        x = self.dropout_layer(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.dropout_layer(x)
        x = self.d3(x)
        return x