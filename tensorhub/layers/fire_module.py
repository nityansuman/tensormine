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


class FireModule(keras.layers.Layer):
    """Building block of `SqueezeNet Model` implemented as a layer. It is mainly used for creating advance features on images.
    For the same accuracy of AlexNet, SqueezeNet can be 3 times faster and 500 times smaller.
    
    More about SqueezeNet: https://arxiv.org/pdf/1602.07360.pdf"""

    def __init__(self, fire_filters=None, skip_connection=False, activation="relu", name=None):
        """Initialize variables.

        Keyword Arguments:
            fire_filters {list} -- List of filters for squeezeing and expanding modules. (default: {[128, 256, 512]})
            skip_connection {bool} -- Boolean to indicate usage of skip-connection in the module.
            activation {str} -- String denoting the activation to be used in the fire module. You can also pass a method for activation. (Pass method as an argument)
            name {str} -- (Optional) Name for the layer.
        """
        if fire_filters is not None:
            assert type(fire_filters) == list and len(fire_filters) == 3 # Requires 3 filter values always
        if name is not None:
            assert type(name) == str
        assert type(skip_connection) == bool # Requires a boolean flag

        super(FireModule, self).__init__(name=name)
        self.fire_filters = [128, 256, 512] if fire_filters == None else fire_filters
        self.skip_connection = skip_connection
        self.act = activation

    def build(self, input_shape):
        """The __call__ method of your layer will automatically run build the first time it is called.
        You now have a layer that's lazy and easy to use.

        Arguments:
            input_shape {tensor} -- Input shape tensor.
        """
        self.squeeze = keras.layers.Conv2D(filters=int(self.fire_filters[0]), kernel_size=(1, 1), strides=(1, 1), activation=None, padding="same", )
        self.expand_1 = keras.layers.Conv2D(filters=int(self.fire_filters[1]), kernel_size=(1, 1), strides=(1, 1),activation=None, padding="same")
        self.expand_2 = keras.layers.Conv2D(filters=int(self.fire_filters[2]), kernel_size=(3, 3), strides=(1, 1), activation=None, padding="same")
        self.batch_norm = keras.layers.BatchNormalization()
        self.activation = keras.layers.Activation(self.act)
        self.concat = keras.layers.Concatenate(axis=-1)
        self.skip_connection = keras.layers.Add()
        self.pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

    def call(self, x):
        """Forward pass of the `Fire Module` layer from SqueezeNet model.

        Arguments:
            x {tensor} -- Input tensor to the layer.

        Returns:
            tensor -- Output tensor from the layer.
        """
        # Keep a copy of inital input for skip connection
        x_ = x
        x = self.squeeze(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        # First block
        x1 = self.expand_1(x)
        x1 = self.batch_norm_1(x1)
        # Second block
        x2 = self.expand_2(x)
        x2 = self.batch_norm_2(x2)
        # Penaltimate block
        x = self.concat([x1, x2])
        # Skip connection if required
        if self.skip_connection:
            x = self.skip_connection_output()([x, x_])
            return self.pool(self.activation(x))
        else:
            return self.activation(x)