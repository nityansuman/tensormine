# Copyright 2021 The TensorHub Authors. All Rights Reserved.
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

# Import packages
import tensorflow as tf


class DotProductAttention(tf.Module):
    """Luong Attention implementation as a modular layer. Most prominantly use in `Neural Machine Translation`."""

    def __init__(self, num_output):
        """Class constructors to initialize input independent variables."""
        super(DotProductAttention, self).__init__()
        self.num_outputs = num_output

    def build(self, input_shape):
        """Initialize input dependent variables.
        
        Arguments:
            input_shape {tensor} -- Input tensor shape.
        """
        self.W1 = self.add_variable("weight1", shape=(int(input_shape[-1]), self.num_outputs))
        self.W2 = self.add_variable("weight2", shape=(int(input_shape[-1]), self.num_outputs))
        self.V = self.add_variable("value", shape=(1, int(input_shape[-1])))

    def call(self, query, value):
        pass