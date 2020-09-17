# Copyright 2020 The TensorHub Authors. All Rights Reserved.
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

# Load package
import setuptools

# Read complete description
with open("README.md", mode="r") as fh:
    long_description = fh.read()


# Create setup
setuptools.setup(
    name="tensorhub",
    version="1.0.0",
    author="Kumar Nityan Suman",
    author_email="nityan.suman@gmail.com",
    description="TensorHub is a library of deep learning models and lego like interlocking blocks designed to make deep learning more accessible and accelerate ML research using TensorFlow 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nityansuman/tensorhub",
    license="MIT",
    packages=setuptools.find_packages(exclude=["examples", ".github", "tests", "metadata"]),
    install_requires=[
        "tensorflow>=2.0.0"
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.1",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)