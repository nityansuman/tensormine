from tensorflow import keras


class SqueezeNet:
    """SqueezeNet is a convolution neural network architecture that produces very small models.
    """

    def __init__(self, n_classes, img_width=256, img_height=256, fire_nodes=None):
        """Class constructor.

        Arguments:
            n_classes {int} -- Number of classes for classification.

        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            num_fire_modules {int} -- Number of fire modules to be used in CNN architecture.
            channels {int} -- number of channels in an input image
        """
        self.n_classes = n_classes
        self.img_height = img_height
        self.img_width = img_width
        self.fire_nodes = [128, 256, 512, 1024] if fire_nodes==None else fire_nodes

    def fire_module(self, name, fire_input, fire_nodes, skip_connection=False):
        """
        This function creates fire module block as per squeezenet network

        Arguments:
            name: Name of the fire module.
            fire_input: Keras input layer to this fire-module block.
            fire_nodes: Nodes representing the outputs intermediate layers in fire module.

        Keyword Arguments:
            skip_connection: weather to merge input with the output of fire-module

        return:
            output layer to current fire-module
        """
        fire_conv = keras.layers.Conv2D(filters=int(fire_nodes[0]), kernel_size=(1, 1), strides=(1, 1),
                                        activation=None, padding="same", name=name+"_conv")(fire_input)
        fire_batch = keras.layers.BatchNormalization(name=name+"_batch")(fire_conv)
        fire_relu = keras.layers.Activation("relu", name=name + "_conv-relu")(fire_batch)

        fire_conv_1 = keras.layers.Conv2D(filters=int(fire_nodes[1]), kernel_size=(1, 1), strides=(1, 1),
                                          activation=None, padding="same", name=name+"_conv1")(fire_relu)
        fire_batch_1 = keras.layers.BatchNormalization(name=name+"_batch1")(fire_conv_1)

        fire_conv_2 = keras.layers.Conv2D(filters=int(fire_nodes[2]), kernel_size=(3, 3), strides=(1, 1),
                                          activation=None, padding="same", name=name+"_conv2")(fire_relu)
        fire_batch_2 = keras.layers.BatchNormalization(name=name+"_batch2")(fire_conv_2)

        fire_penaltimate = keras.layers.Concatenate(name=name+"_concate", axis=-1)([fire_batch_1, fire_batch_2])

        if skip_connection:
            skip_connection_output = keras.layers.Add(name=name+"_skip-connection")([fire_penaltimate, fire_input])
            return keras.layers.Activation("relu", name=name+"_skip_output")(skip_connection_output)
        else:
            return keras.layers.Activation("relu", name=name+"_output")(fire_penaltimate)

    def model(self):
        """Creates squeeze-net CNN architecture.
        Returns:
            keras-model -- Build squeeze-net model with specified number of  fire modules.
        """
        fraction = 0.125
        fire_nodes = self.fire_nodes

        input_tensor = keras.layers.Input(shape=(self.img_width, self.img_height, 3), name="input_layer")
        conv1 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                                    activation=None, padding="same", name="convolution_1")(input_tensor)
        batch1 = keras.layers.BatchNormalization(name="batch_norm_1")(conv1)
        batch1_activated = keras.layers.Activation("relu", name="conv1_relu")(batch1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                          padding="same", name="conv1_maxpooling")(batch1_activated)

        intermediate_layer = pool1
        fire_counter = "0"
        for fire_node in fire_nodes[:-1]:

            fire_counter = str(int(fire_counter) + 1)
            fire_normal = self.fire_module(name="fire"+fire_counter, fire_input=intermediate_layer,
                                           fire_nodes=[fraction*fire_node, int(fire_node/2), int(fire_node/2)],
                                           skip_connection=False)

            fire_counter = str(int(fire_counter) + 1)
            fire_skip = self.fire_module(name="fire"+fire_counter, fire_input=fire_normal,
                                         fire_nodes=[fraction*fire_node, int(fire_node/2), int(fire_node/2)],
                                         skip_connection=True)
            fire_pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                                  name="fire"+fire_counter+"_fire_pool", padding="same")(fire_skip)
            intermediate_layer = fire_pool

        fire_final = self.fire_module(name="fire-final", fire_input=intermediate_layer,
                                      fire_nodes=[fraction * fire_nodes[-1], int(fire_nodes[-1] / 2),
                                                  int(fire_nodes[-1] / 2)],
                                      skip_connection=False)

        conv2 = keras.layers.Conv2D(filters=self.n_classes, kernel_size=(1, 1), strides=(1, 1),
                                    activation=None, padding="same", name="conv_final")(fire_final)
        _, x_dim, y_dim, _ = conv2.get_shape()

        batch2 = keras.layers.BatchNormalization(name="batch_final")(conv2)
        batch2_activated = keras.layers.Activation("relu", name="conv_final_relu")(batch2)
        batch2_activated_pooled = keras.layers.AveragePooling2D(pool_size=(x_dim, y_dim), strides=(1, 1),
                                                                padding="valid", name="final_pool")(batch2_activated)

        # final_output = keras.layers.Reshape((self.n_classes,1), name="final_reshape")(batch2_activated_pooled)
        final_output = keras.layers.Flatten(name="final_reshape")(batch2_activated_pooled)
        model = keras.models.Model(inputs=input_tensor, outputs=final_output)
        return model


if __name__ == "__main__":
    model = SqueezeNet(5).model()
    model.summary()
