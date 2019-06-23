"""
@Author: Kumar Nityan Suman
@Date: 2019-06-23 16:47:38
"""


# Load packages
import numpy as np
from tensorflow import keras


class Embeddings(keras.layers.Layer):
    """Custom embedding class implementation as a keras layer."""

    def __init__(self, input_dim, output_dim=300, max_seq_length=512, learn_embedding=True, embedding_matrix=None):
        """Create an embedding layer for training an embedding or when loading pre-trained embeddings.
        
        Arguments:
            input_dim {int} -- Size of the vocab for the embeddings.
        
        Keyword Arguments:
            output_dim {int} -- Size of the embedding to be learn or pre-trained embedding. (default: {300})
            max_seq_length {int} -- Maximum number of tokens allowed in a sample sequence. (default: {512})
            learn_embedding {bool} -- Flag to set learning embedding as a part of the network. (default: {True})
            embedding_matrix {numpy} -- If 'learn_embedding' is False, use this to load the pre-trained embeddings. (default: {None})        
        """
        assert (learn_embedding == True and embedding_matrix == None) or (learn_embedding == False and embedding_matrix != None)
        if learn_embedding == True:
            self.embedding_layer = keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_seq_length, mask_zero=True)
        elif learn_embedding == False:
            self.embedding_layer = keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, weights=[embedding_matrix], trainable=False, input_length=max_seq_length)

    def call(self, x):
        """Forward pass over the layer.
        
        Returns:
            tensor -- Returns tensor output after embedding lookup.
        """
        x = self.embedding_layer(x)
        return x


class PositionEmbedding(keras.layers.Layer):
    """Positional embedding layer implementation as a keras layer."""
    
    def __init__(self, mode="sum", size=None):
        """Class constructor.
        
        Keyword Arguments:
            mode {str} -- Type of position embedding. (default: {"sum"})
            size {int} -- Size of the position embedding. (default: {None})
        """
        self.size = size
        self.mode = mode
        super(PositionEmbedding, self).__init__()
        
    def call(self, x):
        """Forward pass for positional embeddings.
        
        Arguments:
            x {tensor} -- Input tensor.
        
        Returns:
            tensor -- Returns position embedding tensor.
        """
        # Update position embedding size
        if self.size == None or self.mode == "sum":
            self.size = int(x.shape[-1])
        batch_size = np.shape(x)[0]
        # Compute position j
        position_j = 1. / np.pow(10000., 2 * np.arange(self.size / 2, dtype="float32") / self.size)
        position_j = np.expand_dims(position_j, 0)
        # Compute position i
        position_i = np.cumsum(np.ones_like(x[:,:,0]), 1) -1
        position_i = np.expand_dims(position_i, 2)
        # Compute relative position
        position_ij = np.dot(position_i, position_j)
        position_ij = np.concatenate([np.cos(position_ij), np.sin(position_ij)], 2)
        # Update embedding based on modes
        if self.mode == "sum":
            return position_ij + x
        elif self.mode == "concat":
            return np.concatenate([position_ij, x], 2)
    
    def compute_output_shape(self, input_shape):
        """Compute output shape of the tensor using the input shape.
        
        Arguments:
            input_shape {Tensor} -- Shape tensor.
        
        Returns:
            Tensor -- Output shape tensor.
        """
        # Compute output shape
        if self.mode == "sum":
            return input_shape
        elif self.mode == "concat":
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)