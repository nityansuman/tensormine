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
"""Computer vision models."""

from __future__ import absolute_import, division, print_function

from tensorhub.moap.vision.alex_net import AlexNet
from tensorhub.moap.vision.conv_net import ConvNet
from tensorhub.moap.vision.le_net import LeNet5
from tensorhub.moap.vision.zf_net import ZfNet

__all__ = [
	"LeNet5",
	"ZfNet",
	"ConvNet",
	"AlexNet"
]
