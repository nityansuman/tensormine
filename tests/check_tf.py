import tensorflow as tf

# Installed version
print("TF Version:", tf.__version__)

# Check GPU availability
print("Available GPUs:", tf.config.list_physical_devices("GPU"))