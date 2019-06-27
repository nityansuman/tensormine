""" 
@Author: Kumar Nityan Suman
@Date: 2019-06-25 02:49:11
"""

# Load packages
from tensorflow import keras

class BahdanauAttention(keras.layers.Layer):
    """Bahdanau Attention Implementation as a keras layer."""

    def __init__(self, num_output):
        """Class constructors to initialize input independent variables."""
        super(Attention, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        """Initialize input dependent variables.
        
        Arguments:
            input_shape {tensor} -- Input tensor shape.
        """
        self.W1 = self.add_variable("weight1", shape=(int(input_shape[-1]), self.num_outputs)))
        self.W2 = self.add_variable("weight1", shape=(int(input_shape[-1], self.num_outputs))
        self.V = self.add_variable("weight1", shape=(1, int(input_shape[-1])))

    def call(self, query, value):
        hidden_state = tf.expand_dims(query, 1)

        score = self.V(keras.activations.tanh(tf.matmul(self.W1, value) + tf.matmul(self.W2, hidden_state)))

        # Compute attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Create context vector
        context_vector = tf.reduce_sum(attention_weights * value, axis=1)
        return context_vector