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
from classification_wrapper.model_tail import ModelTail


class VGG16(ModelTail):
    """
    This class is used to create VGG16 architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(VGG16, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create VGG16 architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final VGG16 model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.vgg16.VGG16(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class VGG19(ModelTail):
    """
    This class is used to create VGG19 architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(VGG19, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create VGG19 architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final VGG19 model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.vgg19.VGG19(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class MobileNet(ModelTail):
    """
    This class is used to create MobileNet architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(MobileNet, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create MobileNet architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final MobileNet model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.mobilenet.MobileNet(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class ResNet50(ModelTail):
    """
    This class is used to create ResNet50 architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(ResNet50, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create ResNet50 architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final ResNet50 model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.resnet50.ResNet50(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class InceptionV3(ModelTail):
    """
    This class is used to create InceptionV3 architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=299, img_width=299, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(InceptionV3, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create InceptionV3 architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final InceptionV3 model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.inception_v3.InceptionV3(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class InceptionResNetV2(ModelTail):
    """
    This class is used to create InceptionResNetV2 architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=299, img_width=299, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(InceptionResNetV2, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create InceptionResNetV2 architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final InceptionResNetV2 model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.inception_resnet_v2.InceptionResNetV2(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class Xception(ModelTail):
    """
    This class is used to create Xception architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height, img_width, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(Xception, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create Xception architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final Xception model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.xception.Xception(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class DenseNet121(ModelTail):
    """
    This class is used to create DenseNet121 architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(DenseNet121, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create DenseNet121 architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final DenseNet121 model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.densenet.DenseNet121(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class DenseNet169(ModelTail):
    """
    This class is used to create DenseNet169 architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(DenseNet169, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create DenseNet169 architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final DenseNet169 model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.densenet.DenseNet169(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class DenseNet201(ModelTail):
    """
    This class is used to create DenseNet201 architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(DenseNet201, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create DenseNet201 architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final DenseNet201 model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.densenet.DenseNet201(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class NASNetMobile(ModelTail):
    """
    This class is used to create NASNetMobile architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(NASNetMobile, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create NASNetMobile architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final NASNetMobile model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.nasnet.NASNetMobile(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final


class NASNetLarge(ModelTail):
    """
    This class is used to create NASNetLarge architecture from keras applications modules
    """

    def __init__(self, n_classes, img_height=331, img_width=331, weights="imagenet", num_nodes=None, dropouts=None,
                 activation='relu'):
        """
        Constructor to initialize all model architecture's parameters.
        :param img_height {int}: input height of the image.
        :param img_width {int}: input width of the image.
        :param n_classes {int}: number of classes in the classification tasks.
        :param weights {str}: if "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained
                              must be specified.
        :param num_nodes {list}: list containing dimension(integers) of dense layers.
        :param dropouts {list}: list containing values of dropout(float) layers corresponding to each dense layer.
        :param activation {str}:  activation function to be used for each dense layer.

        Note:
        1. Number of dimensions in num_nodes list will decide, number of dense layers in the top model.
        2. Number of dimension in num_nodes should be equal to number of values in dropouts.
        3. Activation in present in keras should only be specified.
        """
        # parameters initialization
        self.img_height = img_height
        self.img_width = img_width
        self.weights = weights

        # top model parameters initialization
        super(NASNetLarge, self).__init__(n_classes, num_nodes, dropouts, activation)

    @property
    def model(self):
        """
        This function will create NASNetLarge architecture and add sequential model at the top of it for classification.
        :return {keras-model}: final NASNetLarge model ready for classification with specified number of classes.
        """
        # loading the model from keras application module
        self.model = keras.applications.nasnet.NASNetLarge(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.model)
        # creating final classification model by placing the top model over keras model
        model_final = keras.models.Model(inputs=self.model.input, outputs=top_model(self.model.output))
        return model_final
