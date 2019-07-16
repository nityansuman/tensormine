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


class PerceptronClassifier:
    """Multi-layer perceptron based text classifier."""

    def __init__(self, vocab_size, num_classes, max_seq_length=256, num_rnn_layers=2, units=None, dp_rate=None, activation="relu", embedding_dim=100, learn_embedding=True, embedding_matrix=None):
        """Class constructor to initialize member variables.
        
        Arguments:
            vocab_size {int} -- Number of tokens in the vocabulary.
            num_classes {int} -- Number of prediction classes.
        
        Keyword Arguments:
            max_seq_length {int} -- Max. length of an input sequence. (default: {256})
            num_rnn_layers {int} -- Number of stacked hidden layers. (default: {2})
            units {list} -- Number of nodes in each layer. (default: {None})
            dp_rate {list} -- List of `dropout rates` for each hidden layer. If no dropout keep it 0. (default: {0.4})
            embedding_dim {int} -- Size of the embedding to be learned or otherwise. (default: {100})
            learn_embedding {bool} -- Set boolean flag to `True` to learn embedding as part of the neural network. (default: {True})
            embedding_matrix {numpy-array} -- if `learn_embedding` is `False`, use this to load pre-trained embedding vectors. (default: {None})
        """
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.activation = activation
        # Set output activation based on the number of classes
        if self.num_classes == 1:
            self.output_activation = "sigmoid"
        else:
            self.output_activation = "softmax"
        self.max_seq_length = max_seq_length
        self.num_rnn_layers = num_rnn_layers
        self.units = units if units != None else [self.max_seq_length]*(self.num_rnn_layers)
        self.dp_rate = dp_rate if dp_rate != None else [0.3]*(self.num_rnn_layers)
        # Assertion check
        if len(self.units) != self.num_rnn_layers or len(self.dp_rate) != self.num_rnn_layers:
            raise AssertionError("Assertion Error: Length of `units`: {} and `dp_rate`: {} should be same as `num_rnn_layers`: {}".format(len(self.units), len(self.dp_rate), len(self.num_rnn_layers)))
        # Set embeding parameters
        self.learn_embedding = learn_embedding
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix

    def model(self):
        """Create `Perceptron` based text classifier.
        
        Raises:
            ValueError: Raise Error when `learn_embedding` flag and `embedding_matrix` are not in synch.
        
        Returns:
            keras.models.Sequential -- Instance of keras sequential model.
        """
        # Embedding layer
        if self.learn_embedding == True:
            stacked_layers.append(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_seq_length))
        elif self.learn_embedding == False:
            stacked_layers.append(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[self.embedding_matrix], trainable=False, input_length=self.max_seq_length))
        else:
            raise ValueError("Value Error: Wrong Boolean Defined! {}".format(self.learn_embedding, self.embedding_matrix))
        # Stacked hidden layers
        stacked_layers = list()
        for i in range(self.num_rnn_layers):
            stacked_layers.extend([
                keras.layers.Dense(units=self.units[i], activation=self.activation),
                keras.layers.Dropout(rate=self.dp_rate[i])
            ])
        # Output layer
        stacked_layers.append(keras.layers.Dense(units=self.num_classes, activation=self.output_activation))
        model_ins = keras.models.Sequential(stacked_layers)
        return model_ins