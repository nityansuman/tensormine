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


class NER(keras.Model):
    """Model class for RNN based Named entity recognition."""

    def __init__(self, vocab_size, num_classes, num_chars=None, model_name="lstm", max_char_length=25, max_seq_length=256, num_rnn_layers=2, units=None, activation="tanh", char_embedding_dim=50, learn_char_embedding=False, word_embedding_dim=100, learn_word_embedding=True, embedding_matrix=None):
        """Class constructor to initialize member variables.
        
        Arguments:
            vocab_size {int} -- Number of tokens in the vocabulary.            
            num_classes {int} -- Number of prediction classes.
        
        Keyword Arguments:
            model_name {str} -- Name of RNN flavour to use. (default: {"lstm"})
            num_char {int} -- Number of chars in the char vocab.
            max_seq_length {int} -- Max. length of an input word sequence. (default: {256})
            max_char_length {int} -- Max. length of an char sequence in case of char embedding. (default: {25})
            num_rnn_layers {int} -- Number of stacked hidden rnn layers. (default: {2})
            units {list} -- Number of nodes in each layer. (default: {None})
            word_embedding_dim {int} -- Size of the word embedding to be learned or otherwise. (default: {100})
            char_embedding_dim {int} -- Size of the character embedding to be learned or otherwise. (default: {100})
            learn_word_embedding {bool} -- Set boolean flag to `True` to learn word embedding as part of the neural network. (default: {True})
            learn_char_embedding {bool} -- Set boolean flag to `True` to learn character embedding as part of the neural network. (default: {True})
            embedding_matrix {numpy-array} -- if `learn_embedding` is `False`, use this to load pre-trained embedding vectors. (default: {None})
        """
        self.vocab_size = vocab_size
        self.num_chars = num_chars
        self.num_classes = num_classes
        self.model_name = model_name
        self.activation = activation
        # Set output activation based on the number of classes
        if self.num_classes == 1:
            self.output_activation = "sigmoid"
        else:
            self.output_activation = "softmax"

        self.num_rnn_layers = num_rnn_layers
        self.max_char_length = max_char_length
        self.max_seq_length = max_seq_length
        self.units = units if units != None else [self.max_seq_length]*(self.num_rnn_layers)
        # Assertion check
        if len(self.units) != self.num_rnn_layers:
            raise AssertionError("Length of `units`: {} should be same as `num_rnn_layers`: {}".format(len(self.units), len(self.num_rnn_layers)))

        # Set embeding parameters
        self.learn_word_embedding = learn_word_embedding
        self.learn_char_embedding = learn_char_embedding
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.embedding_matrix = embedding_matrix

        # Embedding layers
        if self.learn_word_embedding:
            self.word_emb_layer = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.word_embedding_dim, input_length=self.max_seq_length)
        elif self.learn_word_embedding == False:
            self.word_emb_layer = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.word_embedding_dim, weights=[self.embedding_matrix], trainable=False, input_length=self.max_seq_length)

        if self.learn_char_embedding:
            self.char_emb_layer = keras.layers.TimeDistributed(keras.layers.Embedding(input_dim=self.num_chars, output_dim=self.char_embedding_dim, input_length=self.max_char_length))
            if self.model_name=="lstm":
                self.char_enc_layer = keras.layers.TimeDistributed(keras.layers.LSTM(units=self.char_embedding_dim, return_sequences=False,
                                    recurrent_dropout=0.5))
            else:
                self.char_enc_layer = keras.layers.TimeDistributed(keras.layers.GRU(units=self.char_embedding_dim, return_sequences=False,
                                    recurrent_dropout=0.5))
        if self.model_name=="lstm": 
            self.rnn_layers = [keras.layers.Bidirectional(keras.layers.LSTM(units=self.units[i], activation='tanh',return_sequences=True)) for i in range(self.num_rnn_layers)]
        else:
            self.rnn_layers = [keras.layers.Bidirectional(keras.layers.GRU(units=self.units[i], activation='tanh',return_sequences=True)) for i in range(self.num_rnn_layers)]

        self.dense1 = keras.layers.TimeDistributed((keras.layers.Dense(units=1024, activation="relu")))
        self.dense2 = keras.layers.TimeDistributed((keras.layers.Dense(units=512, activation="relu")))
        self.out = keras.layers.TimeDistributed((keras.layers.Dense(units=self.num_classes, activation='softmax')))
    
    def call(self, inputs):
        """forward pass definition using layers defined above.
        
        Returns:
            keras.Model -- Instance of keras Model.
        """
        if self.learn_char_embedding:
            word_emb = self.word_emb_layer(inputs[0])        
            char_emb = self.char_emb_layer(inputs[1])
            char_enc = self.char_enc_layer(char_emb)
            
            inp = keras.layers.concatenate([word_emb, char_enc])
        else:
            inp = self.word_emb_layer(inputs)
        
        rnn = self.rnn_layers[0](inp) 

        if self.num_rnn_layers > 1:
            for i in range(1,self.num_rnn_layers):
                rnn = self.rnn_layers[i](rnn)           

        dense = self.dense1(rnn)
        dense = self.dense2(dense)
        
        return self.out(dense)             