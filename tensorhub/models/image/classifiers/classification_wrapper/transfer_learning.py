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
from tensorhub.models.image.classifiers.classification_wrapper.model_tail import ModelTail


class VGG16(ModelTail):
    """VGG16 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(VGG16, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.vgg16.VGG16(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model

class VGG19(ModelTail):
    """VGG19 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier. 
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(VGG19, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.vgg19.VGG19(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class MobileNet(ModelTail):
    """MobileNet based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(MobileNet, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.mobilenet.MobileNet(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class ResNet50(ModelTail):
    """ResNet50 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(ResNet50, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.resnet50.ResNet50(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class InceptionV3(ModelTail):
    """InceptionV3 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=299, img_width=299, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(InceptionV3, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.inception_v3.InceptionV3(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class InceptionResNetV2(ModelTail):
    """InceptionResNetV2 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=299, img_width=299, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(InceptionResNetV2, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class Xception(ModelTail):
    """XceptionNet based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height, img_width, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(Xception, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.xception.Xception(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class DenseNet121(ModelTail):
    """DenseNet121 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(DenseNet121, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.densenet.DenseNet121(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class DenseNet169(ModelTail):
    """DenseNet169 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(DenseNet169, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.densenet.DenseNet169(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class DenseNet201(ModelTail):
    """DenseNet201 based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(DenseNet201, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.densenet.DenseNet201(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class NASNetMobile(ModelTail):
    """NASNetMobile based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=224, img_width=224, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(NASNetMobile, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.nasnet.NASNetMobile(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model


class NASNetLarge(ModelTail):
    """NASNet Large based image classification model with transfer learning support on imagenet weights.
    
    Arguments:
        ModelTail {cls} -- Template class to convert base architetcure to classifier.    
    """

    def __init__(self, n_classes, img_height=331, img_width=331, weights="imagenet", num_nodes=None, dropouts=None, activation="relu"):
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
        self.weights = weights
        # Initiate base model architecture
        super(NASNetLarge, self).__init__(n_classes, num_nodes, dropouts, activation)

    def model(self):
        """Create image classifier.

        Returns:
            keras-model -- Model for image classification with specified configuration.
        """
        # Load base model using keras application module
        self.base_model = keras.applications.nasnet.NASNetLarge(
            weights=self.weights,
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Creating top sequential model as per specified parameters
        top_model = self.create_model_tail(self.base_model)
        # Stich to create classification model
        model = keras.models.Model(inputs=self.base_model.input, outputs=top_model(self.base_model.output))
        return model