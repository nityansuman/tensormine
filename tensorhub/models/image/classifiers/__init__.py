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

# Import modules

# SOTA models support transfer learning
from tensorhub.models.image.classifiers.transfer_learning import VGG16, VGG19
from tensorhub.models.image.classifiers.transfer_learning import InceptionV3, InceptionResNetV2
from tensorhub.models.image.classifiers.transfer_learning import MobileNet
from tensorhub.models.image.classifiers.transfer_learning import NASNetMobile, NASNetLarge
from tensorhub.models.image.classifiers.transfer_learning import ResNet50
from tensorhub.models.image.classifiers.transfer_learning import Xception
from tensorhub.models.image.classifiers.transfer_learning import DenseNet121, DenseNet169, DenseNet201

# Basic models
from tensorhub.models.image.classifiers.basic_classifiers import SmallVGG

# Advance models
from tensorhub.models.image.classifiers.inception_v4 import InceptionV4