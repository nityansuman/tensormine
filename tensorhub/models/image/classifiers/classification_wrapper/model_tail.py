
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


class ModelTail:
    """Create a image classifier model using the a pre-defined base model."""

    def __init__(self, n_classes, num_nodes=None, dropouts=None, activation="relu"):
        """Constructor to initialize model parameters.
        
        Arguments:
            n_classes {int} -- Number of classes.
            num_nodes {list} -- List of number of nodes in the dense layers. It also decides number of dense layers.
            dropouts {list} -- List containing dropout rate to each dense layer.
            activation {str} -- Activation function to be used for each dense layer.
        """
        # parameters initialization
        self.n_classes = n_classes
        if self.n_classes == 1:
            self.output_act = "sigmoid"
        else:
            self.output_act = "softmax"
        self.num_nodes = num_nodes if num_nodes != None else [1024, 512]
        self.dropouts = dropouts if dropouts != None else [0.5, 0.5]
        self.activation = activation

        # Check if number of layers and number of dropouts have same dimension
        if not len(self.num_nodes) == len(self.dropouts):
            raise AssertionError()

    def create_model_tail(self, model):
        """Method creates top model. This model will be added at the top of keras application model.

        Arguments:
            model {keras-model} -- input keras application model.
    
        Returns:
            sequential model with dense layers, dropout layers and softmax layer as specified.
        """
        # Creating a sequential model to at as top layers
        top_model = keras.Sequential()
        top_model.add(keras.layers.Flatten(input_shape=model.output_shape[1:]))

        # Add multiple layers
        for layer_num, layer_dim in enumerate(self.num_nodes):
            top_model.add(keras.layers.Dense(layer_dim, activation=self.activation))
            top_model.add(keras.layers.Dropout(self.dropouts[layer_num]))
        
        top_model.add(keras.layers.Dense(self.n_classes, activation=self.output_act))
        return top_model
