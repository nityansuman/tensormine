""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-28 00:46:39
@Last Modified Time: 2019-05-28 00:46:39
"""

# Load packages
import os
import sys

from models import *


class SequenceClassification():
    def __init__(self):
        pass

    def get_simple_rnn(self, vocab_size, num_classes, max_length=512, num_nodes=512, output_activation="softmax", activation=None, learn_embedding=True, embedding_matrix=None):
        return SimpleRNN(max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding, embedding_matrix)

    def get_simple_lstm(self, vocab_size, num_classes, max_length=512, num_nodes=512, output_activation="softmax", activation=None, learn_embedding=True, embedding_matrix=None):
        return SimpleLSTM(vocab_size, num_classes, max_length, num_nodes, activation, output_activation, learn_embedding, embedding_matrix)
        
    def get_simple_gru(self, vocab_size, num_classes, max_length=512, num_nodes=512, output_activation="softmax", activation=None, learn_embedding=True, embedding_matrix=None):
        return SimpleGRU(max_length, vocab_size, num_classes, num_nodes, activation, learn_embedding, embedding_matrix)

    def get_text_cnn(self, vocab_size, num_classes, max_length=512, num_nodes=1024, num_filter=128, kernal_size=3, stride=1, dropout_rate=0.4, activation="relu", output_activation="softmax", learn_embedding=True, embedding_matrix=None):
        return TextCNN(vocab_size, num_classes, max_length, num_nodes, num_filter, kernal_size, stride, dropout_rate, activation, output_activation, learn_embedding, embedding_matrix)