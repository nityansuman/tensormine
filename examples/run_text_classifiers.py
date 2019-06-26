"""
@Author: Kumar Nityan Suman
@Date: 2019-05-28 02:56:59

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

# Load packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorhub.text.wrapper.embeddings import Embeddings
from tensorhub.text.cooked_models.classification import SimpleTextClassification

# Multiclass Text Classification on 'News Healines' Dataset

# DATA PREPRATION
#
#
# Path to dataset in use
filepath = "~/__data__/news-category.json"

# Load data into a dataframe
df = pd.read_json(filepath, orient="records", encoding="utf-8", lines=True)

print("Data Shape:", df.shape)
print("Columns:", df.columns)

# Select feature and target column
x = list(df.headline)
y = list(df.category)

# Split data into train and test
# Train -> 75%
# Test -> 25%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print("Number of train samples:", len(x_train)) # List
print("Number of test samples:", len(x_test))

# Compute the maximum number of words and characters in a sequence from the whole corpus
max_num_words = len(max(x).split())
max_num_chars = len(max(x))

print("Maximum number of word in a sequence are", max_num_words)
print("Maximum number of chars in a sequence are", max_num_chars)

# Train a custom tokenier on the corpus and generate tokenizer instance with vocabulary
# type_embedding: 'word' & 'char'
tokenizer, word_index = Embeddings.create_vocabulary(x, type_embedding="word")

# Load pre-trained embedding matrix if learn_embedding is False. See model configuration.
# embedding_matrix = Embeddings.load_embedding(word_index, filepath="‎⁨~/__embeddings__⁩/glove.840B.300d.txt", dim=300)

# Tokenize data using the created tokenizer instance
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# Pad or truncate sequences to make fixed length input
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train,
    value=0, # Pad with 0
    padding="post",
    truncating="post",
    maxlen=max_num_words # When using 'char' embedding use max_num_chars
)

x_test = keras.preprocessing.sequence.pad_sequences(
    x_test,
    value=0, # Pad with 0
    padding="post",
    truncating="post",
    maxlen=max_num_words # When using 'char' embedding use max_num_chars
)

# Convert data type to float from int
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Get all class labels
classes = np.unique(y_train)

# Generate a unique index for each class from 0 to len(classes) - 1
index = range(len(classes))

# Create class to index mapping
class_index = dict(zip(classes, index))
reverse_class_index = dict(zip(index, classes))

# Now encode labels using the created mapping
y_train = [class_index[label] for label in y_train]
y_test = [class_index[label] for label in y_test]

# Convert to categorical
y_train = keras.utils.to_categorical(y_train, num_classes=len(classes))
y_test = keras.utils.to_categorical(y_test, num_classes=len(classes))

# Create batch datasets: batches of 32 for train and 64 for test
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

# LOAD ONE OF THE COOKED MODELS FROM OUR LIBRARY
#
#
# GRU model
custom_model = SimpleTextClassification(
    model_name="gru",
    vocab_size=len(word_index) + 1,
    num_classes=len(classes)
)

# SET MODEL TRAINING AND VALIDATION ENVIRONMENT
#
#
# Train model using a tensor function
# In future, this will be separated as a different module with multiple development and production environment
def train_validate_model(model, train_ds, test_ds, epochs=3):
    # Define model configuration
    loss_function = keras.losses.CategoricalCrossentropy()

    # Define optimizer
    optimizer = keras.optimizers.RMSprop()

    # Accumulate performance metrics while training
    train_loss = keras.metrics.Mean(name="train_loss")
    train_accuracy = keras.metrics.CategoricalAccuracy(name="train_accuracy")
    test_loss = keras.metrics.Mean(name="test_loss")
    test_accuracy = keras.metrics.CategoricalAccuracy(name="test_accuracy")
    template = "Epoch {}, Loss: {}, Accuracy: {}%, Test Loss: {}, Test Accuracy: {}%"

    @tf.function()
    def train_step(text, labels):
        # Use gradient tape for training the model
        with tf.GradientTape() as tape:
            # Get predictions
            predictions = model(text)
            # Compute instantaneous loss
            loss = loss_function(labels, predictions)
        # Update gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Store
        train_loss(loss)
        train_accuracy(labels, predictions)
        
    # Test model using another tensor function
    @tf.function()
    def test_step(text, labels):
        # Get predictions
        predictions = model(text)
        # Compute instantaneous loss
        loss = loss_function(labels, predictions)
        # Store
        test_loss(loss)
        test_accuracy(labels, predictions)

    print("{:#^50s}".format("Train and Validate"))
    for epoch in range(1, epochs+1):
        print("Epoch {}/{}".format(epoch, epochs))
        
        # Train model on batches
        for text, labels in train_ds:
            train_step(text, labels)
        
        # Test model on batches
        for t_text, t_labels in test_ds:
            test_step(t_text, t_labels)

        # Prompt user
        print(template.format(epoch, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))

# Run
train_validate_model(custom_model, train_ds, test_ds)