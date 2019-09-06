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


class VGG16(ModelTail, keras.models.Model):
    """VGG16 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights_ {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.vgg16.VGG16(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class VGG19(ModelTail, keras.models.Model):
    """VGG19 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier. 
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights_ {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.vgg19.VGG19(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class MobileNet(ModelTail, keras.models.Model):
    """MobileNet based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights_ {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.mobilenet.MobileNet(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class ResNet50(ModelTail, keras.models.Model):
    """ResNet50 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights_ {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.resnet50.ResNet50(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class InceptionV3(ModelTail, keras.models.Model):
    """InceptionV3 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=299, img_width=299, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights_ {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.inception_v3.InceptionV3(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class InceptionResNetV2(ModelTail, keras.models.Model):
    """InceptionResNetV2 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=299, img_width=299, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights_ {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class Xception(ModelTail, keras.models.Model):
    """XceptionNet based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height, img_width, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.xception.Xception(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class DenseNet121(ModelTail, keras.models.Model):
    """DenseNet121 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights_ {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.densenet.DenseNet121(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class DenseNet169(ModelTail, keras.models.Model):
    """DenseNet169 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.densenet.DenseNet169(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class DenseNet201(ModelTail, keras.models.Model):
    """DenseNet201 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.densenet.DenseNet201(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class NASNetMobile(ModelTail, keras.models.Model):
    """NASNetMobile based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.nasnet.NASNetMobile(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)


class NASNetLarge(ModelTail, keras.models.Model):
    """NASNet Large based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=331, img_width=331, weights_="imagenet", num_nodes=None, dropouts=None, activation="relu"):
        """Class constructor.
        
        Arguments:
            n_classes {int} -- Number of classes for classification.
    
        Keyword Arguments:
            img_height {int} -- Height of the input image.
            img_width {int} -- Width of the input image.
            weights {str} -- If "imagenet" pre-trained imagenet weights will be downloaded. Else path to custom trained weights must be specified.
            num_nodes {list} -- List of nodes for each dense layer.
            dropouts {list} -- List of dropout rate corresponding to each dense layer.
            activation {str} --  Activation to be used for each dense layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.weights_ = weights_
        # Initiate base model architecture and keras.models.Model
        ModelTail.__init__(self, n_classes, num_nodes, dropouts)
        keras.models.Model.__init__(self)

        # Load base model using keras application module
        self.base_model = keras.applications.nasnet.NASNetLarge(
            weights=self.weights_,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        self.top_model = self.create_model_tail(self.base_model)

    def call(self, x):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """

        # Stich to create classification model
        y = self.base_model(x)
        return self.top_model(y)