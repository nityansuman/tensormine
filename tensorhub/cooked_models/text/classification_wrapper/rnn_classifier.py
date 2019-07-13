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


class RNNClassifier:
    """Sequence classification with different flavours of RNNs."""

    def __init__(self, vocab_size, num_classes, model_name="lstm", bidir=False, max_seq_length=256, num_rnn_layers=2, units=None, activation="tanh", embedding_dim=100, learn_embedding=True, embedding_matrix=None):
        """Class constructor to initialize member variables.
        
        Arguments:
            vocab_size {int} -- Number of tokens in the vocabulary.
            num_classes {int} -- Number of prediction classes.
        
        Keyword Arguments:
            model_name {str} -- Name of RNN flavour to use. (default: {"lstm"})
            bidir {bool} -- Set boolean flag to use bidirectional RNNs. (default: {False})
            max_seq_length {int} -- Max. length of an input sequence. (default: {256})
            num_rnn_layers {int} -- Number of stacked hidden rnn layers. (default: {2})
            units {list} -- Number of nodes in each layer. (default: {None})
            embedding_dim {int} -- Size of the embedding to be learned or otherwise. (default: {100})
            learn_embedding {bool} -- Set boolean flag to `True` to learn embedding as part of the neural network. (default: {True})
            embedding_matrix {numpy-array} -- if `learn_embedding` is `False`, use this to load pre-trained embedding vectors. (default: {None})
        """
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.model_name = model_name
        self.activation = activation
        # Set output activation based on the number of classes
        if self.num_classes == 1:
            self.output_activation = "sigmoid"
        else:
            self.output_activation = "softmax"
        self.bidir = bidir
        self.num_rnn_layers = num_rnn_layers
        self.max_seq_length = max_seq_length
        self.units = units if units != None else [self.max_seq_length]*(self.num_rnn_layers)
        # Assertion check
        if len(self.units) != self.num_rnn_layers:
            raise AssertionError("Length of `units`: {} should be same as `num_rnn_layers`: {}".format(len(self.units), len(self.num_rnn_layers)))
        # Set embeding parameters
        self.learn_embedding = learn_embedding
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
    
    def model(self):
        """Create `RNN` based text classifier.
        
        Raises:
            ValueError: Raise Error when `learn_embedding` flag and `embedding_matrix` are not in synch.
        
        Returns:
            keras.models.Sequential -- Instance of keras sequential model.
        """
        stacked_layers = list()
        # Embedding layer
        if self.learn_embedding == True:
            stacked_layers.append(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_seq_length))
        elif self.learn_embedding == False:
            stacked_layers.append(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[self.embedding_matrix], trainable=False, input_length=self.max_seq_length))
        # Stacked rnn hidden layer
        if self.bidir == False:
            # Unidirectional rnn layers
            if self.model_name == "lstm":
                for i in range(self.num_rnn_layers):
                    if i == self.num_rnn_layers - 1:
                        stacked_layers.append(keras.layers.LSTM(units=self.units[i], activation=self.activation))
                    else:
                        stacked_layers.append(keras.layers.LSTM(units=self.units[i], activation=self.activation, return_sequences=True))
            elif self.model_name == "gru":
                for i in range(self.num_rnn_layers):
                    if i == self.num_rnn_layers - 1:
                        stacked_layers.append(keras.layers.GRU(units=self.units[i], activation=self.activation))
                    else:
                        stacked_layers.append(keras.layers.GRU(units=self.units[i], activation=self.activation, return_sequences=True))
            elif self.model_name == "rnn":
                for i in range(self.num_rnn_layers):
                    if i == self.num_rnn_layers - 1:
                        stacked_layers.append(keras.layers.RNN(units=self.units[i], activation=self.activation))
                    else:
                        stacked_layers.append(keras.layers.RNN(units=self.units[i], activation=self.activation, return_sequences=True))
            else:
                raise ValueError("Wrong rnn layer passed: {}".format(self.model_name))
        else:
            # Bidirectional rnn layers
            if self.model_name == "lstm":
                for i in range(self.num_rnn_layers):
                    if i == self.num_rnn_layers - 1:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.LSTM(units=self.units[i], activation=self.activation)))
                    else:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.LSTM(units=self.units[i], activation=self.activation, return_sequences=True)))
            elif self.model_name == "gru":
                for i in range(self.num_rnn_layers):
                    if i == self.num_rnn_layers - 1:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.GRU(units=self.units[i], activation=self.activation)))
                    else:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.GRU(units=self.units[i], activation=self.activation, return_sequences=True)))
            elif self.model_name == "rnn":
                for i in range(self.num_rnn_layers):
                    if i == self.num_rnn_layers - 1:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.RNN(units=self.units[i], activation=self.activation)))
                    else:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.RNN(units=self.units[i], activation=self.activation, return_sequences=True)))
            else:
                raise ValueError("Wrong rnn layer passed: {}".format(self.model_name))
        # Classifier layers
        stacked_layers.extend([
            keras.layers.Dropout(rate=0.4),
            keras.layers.Dense(units=1024, activation="relu"),
            keras.layers.Dense(units=1024, activation="relu"),
            keras.layers.Dense(units=self.num_classes, activation=self.output_activation)
        ])
        model_ins = keras.Sequential(stacked_layers)
        return model_ins