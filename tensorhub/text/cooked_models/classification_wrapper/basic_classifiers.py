""" 
@Author: Kumar Nityan Suman
@Date: 2019-06-26 23:12:13

Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Load packages
from tensorflow import keras
from wrapper.embeddings import EmbeddingLayer


class PerceptronClassifier:
    """Multi-layer perceptron based text classifier."""
    def __init__(self, vocab_size, num_classes, max_seq_length=256, num_layers=2, units=None, dp_rate=0.4, embedding_dim=100, learn_embedding=True, embedding_matrix=None):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        if self.num_classes == 1:
            self.output_activation = "sigmoid"
        else:
            self.output_activation = "softmax"
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.default_units = [256]*(self.num_layers)
        self.units = units if units != None else self.default_units
        self.dp_rate = dp_rate
        self.learn_embedding = learn_embedding
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix

    @property
    def model(self):
        stacked_layers = list()
        # Embedding layer
        if self.learn_embedding == True:
            stacked_layers.append(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_seq_length))
        elif self.learn_embedding == False:
            stacked_layers.append(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[self.embedding_matrix], trainable=False, input_length=self.max_seq_length))
        # Perceptron layer
        for i in range(self.num_layers + 1):
            if i == self.num_layers:
                stacked_layers.extend([
                    keras.layers.Dropout(rate=self.dp_rate),
                    keras.layers.Dense(units=self.num_classes, activation=self.output_activation)
                ])
            else:
                stacked_layers.append(keras.layers.Dense(units=self.units[i], activation="relu"))
        model = keras.models.Sequential(stacked_layers)
        return model


class SequenceClassifier:
    """Sequence classification with different flavours of RNNs."""
    def __init__(self, vocab_size, num_classes, model_name="lstm", max_seq_length=256, bidir=False, num_layers=2, units=None, dp_rate=0.4, embedding_dim=100, learn_embedding=True, embedding_matrix=None):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.model_name = model_name
        if self.num_classes == 1:
            self.output_activation = "sigmoid"
        else:
            self.output_activation = "softmax"
        self.bidir = bidir
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.default_units = [128]*(self.num_layers)
        self.dp_rate = dp_rate
        self.units = units if units != None else self.default_units
        self.learn_embedding = learn_embedding
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix

    @property
    def model(self):
        stacked_layers = list()
        # Embedding layer
        if self.learn_embedding == True:
            stacked_layers.append(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_seq_length))
        elif self.learn_embedding == False:
            stacked_layers.append(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[self.embedding_matrix], trainable=False, input_length=self.max_seq_length))
        # Main logic layer
        if self.bidir == False:
            if self.model_name == "lstm":
                for i in range(self.num_layers):
                    if i == self.num_layers - 1:
                        stacked_layers.append(keras.layers.LSTM(units=self.units[i]))
                    else:
                        stacked_layers.append(keras.layers.LSTM(units=self.units[i], return_sequences=True))
            elif self.model_name == "gru":
                for i in range(self.num_layers):
                    if i == self.num_layers - 1:
                        stacked_layers.append(keras.layers.GRU(units=self.units[i]))
                    else:
                        stacked_layers.append(keras.layers.GRU(units=self.units[i], activation="tanh", return_sequences=True))
            else:
                for i in range(self.num_layers):
                    if i == self.num_layers - 1:
                        stacked_layers.append(keras.layers.RNN(units=self.units[i]))
                    else:
                        stacked_layers.append(keras.layers.RNN(units=self.units[i], return_sequences=True))
        else:
            if self.model_name == "lstm":
                for i in range(self.num_layers):
                    if i == self.num_layers - 1:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.LSTM(units=self.units[i])))
                    else:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.LSTM(units=self.units[i], return_sequences=True)))
            elif self.model_name == "gru":
                for i in range(self.num_layers):
                    if i == self.num_layers - 1:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.GRU(units=self.units[i])))
                    else:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.GRU(units=self.units[i],return_sequences=True)))
            else:
                for i in range(self.num_layers):
                    if i == self.num_layers - 1:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.RNN(units=self.units[i])))
                    else:
                        stacked_layers.append(keras.layers.Bidirectional(keras.layers.RNN(units=self.units[i],return_sequences=True)))
        # Classifier layers
        stacked_layers.extend([
            keras.layers.Dropout(rate=self.dp_rate),
            keras.layers.Dense(units=1024, activation="relu"),
            keras.layers.Dense(units=1024, activation="relu"),
            keras.layers.Dense(units=self.num_classes, activation=self.output_activation)
        ])
        model = keras.Sequential(stacked_layers)
        return model
