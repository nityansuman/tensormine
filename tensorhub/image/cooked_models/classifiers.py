"""
@Author: Kumar Nityan Suman
@Date: 2019-06-27 04:29:11

Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Load modules
from classification_wrapper.basic_classifiers import CNNClassifier

from classification_wrapper.transfer_learning import VGG16
from classification_wrapper.transfer_learning import VGG19
from classification_wrapper.transfer_learning import MobileNet
from classification_wrapper.transfer_learning import ResNet50
from classification_wrapper.transfer_learning import InceptionV3
from classification_wrapper.transfer_learning import InceptionResNetV2
from classification_wrapper.transfer_learning import Xception
from classification_wrapper.transfer_learning import DenseNet121
from classification_wrapper.transfer_learning import DenseNet169
from classification_wrapper.transfer_learning import DenseNet201
from classification_wrapper.transfer_learning import NASNetMobile
from classification_wrapper.transfer_learning import NASNetLarge


