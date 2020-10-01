import sys
sys.path.append("..")

import tensorflow as tf
from tensorhub.moap.image import ConvNet


if __name__ == "__main__":
	import numpy as np
	# Model / data parameters
	num_classes = 10
	input_shape = (28, 28, 1)

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	# Scale images to the [0, 1] range
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255
	# Make sure images have shape (28, 28, 1)
	x_train = np.expand_dims(x_train, -1)
	x_test = np.expand_dims(x_test, -1)

	print("x_train shape:", x_train.shape)
	print(x_train.shape[0], "train samples")
	print(x_test.shape[0], "test samples")

	# convert class vectors to binary class matrices
	y_train = tf.keras.utils.to_categorical(y_train, num_classes)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes)

	# Load model architecture
	model = ConvNet(num_classes=10, name="ConvNet")
	
	# Compile model
	model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
	
	# Build model with input shape
	model.build(input_shape=(None, 28, 28, 1))
	
	# Get model summary
	print(model.summary())