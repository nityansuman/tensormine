"""
@Author: Kumar Nityan Suman
@Date: 2019-05-28 01:37:20
"""


# Load packages
from tensorflow import keras

# Import TensorMine Lego Block
from text.blocks.embeddings import Embeddings


class SimpleTextClassification(keras.Model):
    """Simple text classification model implementation using recurrent neural networks."""

    def __init__(self, vocab_size, num_classes, model_name="lstm", bidir=False, max_seq_length=256, num_nodes=None, learn_embedding=True, embedding_dim=300, embedding_matrix=None):
        """Text classification using some basic standard recurrent neural network architectures.
        
        Arguments:
            vocab_size {int} -- Integer denoting number of top words to consider for the vocab.
            num_classes {int} -- Number of classes.
        
        Keyword Arguments:
            model_name {str} -- Name of the recurrent layer to use. Choices = ['lstm', 'gru'] (default: {'lstm'})
            bidir {bool} -- Boolean to represent if bi-directional rnn layer is to be used.
            max_seq_length {int} -- Maximum length of the input sequence length. (default: {256})
            num_nodes {list} -- Number of nodes in rnn and otther two dense layers respectively. (default: {[256, 512]})
            learn_embedding {bool} -- Boolean denoting to learn embedding or use pre-trained embeddings. (default: {True})
            embedding_dim {int} -- Embedding dimension. (default: {300})
            embedding_matrix {numpy} -- Pre-trained embedding loaded into a numpy matrix if 'learn_embedding' == True. (default: {None})
        """
        super(SimpleTextClassification, self).__init__()
        self.embedding = Embeddings.EmbeddingLayer(vocab_size=vocab_size, max_seq_length=max_seq_length, embedding_dim=embedding_dim, learn_embedding=learn_embedding, embedding_matrix=embedding_matrix)
        self.num_nodes = num_nodes if num_nodes != None else [256, 512]
        # Recurrent layer
        if bidir == False:
            if model_name == "lstm":
                self.main_layer = keras.layers.LSTM(units=num_nodes[0])
            elif model_name == "gru":
                self.main_layer = keras.layers.GRU(units=num_nodes[0])
        else:
            if model_name == "lstm":
                self.main_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=num_nodes[0]))
            elif model_name == "gru":
                self.main_layer = keras.layers.Bidirectional(keras.layers.GRU(units=num_nodes[0]))
        # Dense layer
        self.dense_layer1 = keras.layers.Dense(units=self.num_nodes[1], activation="relu")
        self.dense_layer2 = keras.layers.Dense(units=self.num_nodes[1], activation="relu")
        # Final output layer with softmax/sigmoid
        if num_classes > 1:
            self.output_layer = keras.layers.Dense(units=num_classes, activation="softmax")
        else:
            self.output_layer = keras.layers.Dense(units=1, activation="sigmoid")

    
    def call(self, x):
        """Forward pass over the network.
        
        Arguments:
            x {Tensor} -- Input tensor in the forward pass.
        
        Returns:
            numpy -- Output tensor.
        """
        x = self.embedding(x)
        x = self.main_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.output_layer(x)
        return x


class TextCNN(keras.Model):
    """Text classification using 2D convolutional neural networks."""

    def __init__(self, vocab_size, num_classes, max_length=512, filters=None, kernals=None, strides=None, drop_rate=0.4, learn_embedding=True, embedding_dim=300, embedding_matrix=None):
        super(keras.Model, self).__init__()
        self.filters = filters if filters != None else [128, 128]
        self.kernals = kernals if kernals != None else [5, 5]
        self.strides = strides if strides != None else [2, 2]
        # Embedding
        self.embedding = Embeddings.EmbeddingLayer(vocab_size=vocab_size, max_seq_length=max_seq_length, embedding_dim=embedding_dim, learn_embedding=learn_embedding, embedding_matrix=embedding_matrix)
        self.reshape_layer = keras.layers.Reshape((max_length, embed_dim, 1))
        self.conv_layer1 = keras.layers.Conv2D(filters=self.filters[0], kernel_size=(self.kernals[0]), activation=activation)
        self.pool_layer1 = keras.layers.MaxPool2D(pool_size=max_length-filters[0]+1, strides=(self.strides[0]))
        self.conv_layer2 = keras.layers.Conv2D(filters=self.filters[1], kernel_size=(self.kernals[1]), activation=activation)
        self.pool_layer2 = keras.layers.MaxPool2D(pool_size=max_length-filters[1]+1, strides=(self.strides[1]))
        self.flatten_layer = keras.layers.Flatten()
        self.dropout_layer = keras.layers.Dropout(rate=drop_rate)
        self.output_layer = keras.layers.Dense(units=num_classes, activation=output_activation)
    
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = self.reshape_layer(x)
        x = self.conv_layer1(x)
        x = self.pool_layer1(x)
        x = self.conv_layer2(x)
        x = self.pool_layer2(x)
        x = self.flatten_layer(x)
        x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x