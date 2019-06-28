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
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential


class ModelTail:
    """
    This class is used to add dense layers, dropout layers and softmax layer at the top of keras application model.
    """
    def __init__(self, n_classes, num_nodes=None, dropouts=None, activation='relu'):
        """
        Constructor to initialize top model architecture's parameters.
        :param n_classes {int}: number of classes in the classification tasks.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}: activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.n_classes = n_classes
        self.num_nodes = num_nodes if num_nodes != None else [1024,512]
        self.dropouts = dropouts if dropouts != None else [0.5,0.5]
        self.activation = activation

        # check if the length of "list of dense-layer dimensions" and "list of dropout values" is same
        if (num_nodes != None): assert len(num_nodes) == len(self.dropouts)

    def create_model_tail(self, model):
        """
        This function is used to create top model. This model will be added at the top of keras application model.

        :param model {keras-model}: input keras application model.
        :return: sequential model with dense layers, dropout layers and softmax layer as specified.
        """
        # creating a sequential model to at as top layers
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        for layer_num, layer_dim in enumerate(self.num_nodes):
            top_model.add(Dense(layer_dim, activation=self.activation))
            top_model.add(Dropout(self.dropouts[layer_num]))
        top_model.add(Dense(self.n_classes, activation='softmax'))
        return top_model