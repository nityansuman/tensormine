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


def create_vocabulary(corpus, type_embedding="word", num_words=10000):
    """Create a sequence tokenizer and generate vocabulary. Supports both 'word' and 'char' sequences.
    
    Arguments:
        corpus {list} -- A list of strings from which word-index or char-index mapping is created.
        num_words {int} -- Maximum number of words to keep, based on word frequency. \
            Only the most common (num_words-1) tokens will be kept. Not necessary when doing character embedding.
    
    Returns:
        TokenizerObject -- Tokenizer object to fit sequences.
        Dict -- Vocabulary dictionary.
    """
    # Custom tokenizer
    if type_embedding.lower() == "word":
        # Word embeddings
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<UNK>")
    else:
        # Character embeddings
        tokenizer = keras.preprocessing.text.Tokenizer(char_level=True, oov_token="<UNK>")
    # Fit tokenizer on the corpus
    tokenizer.fit_on_texts(corpus)
    # Generate vocabulary
    vocab = tokenizer.word_index
    return tokenizer, vocab

def load_embedding(filepath, token_index_mapping, embedding_dim=300):
    """Create an embedding matrix from the given pre-trained vector.
    
    Arguments:
        filepath {str} -- Path to load pre-trained embeddings (ex: glove).
        embedding_dim {int} -- Dimension of the pre-trained embedding.
        token_index_mapping {dict} -- A dictionary containing token-index mapping from the whole corpus.
    
    Returns:
        Matrix -- A numpy matrix containing embeddings for each token in the token-index mapping.
    """
    # Placeholder for embedding
    embedding_index = dict()
    # Access file to load pre-trained embedding
    with open(filepath, mode="r") as fp:
        for line in fp:
            values = line.split()
            token = values[0:-dim]
            coefs = values[-dim:]
            embedding_index[token[0]] = coefs
    # Create a weight matrix for token in training docs
    embedding_matrix = np.zeros((len(token_index_mapping), embedding_dim))
    # Create token-index mapping
    for token, i in word_index.items():
        embedding_vector = embeddings_index.get(token)
        # Update embedding
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix