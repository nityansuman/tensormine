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
    description="TensorHub is a library built on top of TensorFlow 2 to provide simple, modular and repeatable abstractions to accelerate deep learning research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nityansuman/tensorhub",
    license="Apache-2.0",
    packages=setuptools.find_packages(exclude=[".github", "tests", "metadata", "examples", "docs"]),
    install_requires=[
        "tensorflow==2.4.0"
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)