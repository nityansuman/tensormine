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
import numpy as np


def accuracy(y_true, y_pred):
    """Calculates how often predictions matches labels.
    For example, if `y_true` is [1, 2, 3, 4] and `y_pred` is [0, 2, 3, 4] then the accuracy is 3/4 or 0.75.
    
    Arguments:
        y_true {list} -- Original values.
        y_pred {list} -- Predicted valued.
    
    Returns:
        float -- Accuracy computed on the given pair of values.
    """
    matches = [i for i, j in zip(y_true, y_pred) if i == j]
    total = len(y_true)
    return matches / total

def mean(values):
    """Computes the (weighted) mean of the given values.

    Arguments:
        values {list} -- A list of values.
    
    Returns:
        float -- Mean computed on the values.
    """
    count = len(values)
    total = np.sum(values)
    return total / count