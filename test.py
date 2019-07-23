# Test

# Import image models
from tensorhub.models.image.classifiers import InceptionResNetV2, VGG16, SmallVGG

# Import text model
from tensorhub.models.text.classifiers import PerceptronClassifier, LSTMClassifier, GRUClassifier

# Import utilities
from tensorhub.utilities.activations import relu, gelu, softmax
from tensorhub.utilities.processor import create_vocabulary, load_embedding

from tensorhub.layers import BertLayer, BertConfig
import tensorflow as tf
print("Testing Bert Layer")
max_seq_len = 128

# bert config
init_checkpoint = "/Users/naval/Desktop/Naval/CBM/BERT-NER-master/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt"
bert_config_file = "/Users/naval/Desktop/Naval/CBM/BERT-NER-master/bert/multi_cased_L-12_H-768_A-12/bert_config.json"
bert_config = BertConfig.from_json_file(bert_config_file)

# fake input for initializing weights
input_ids      = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
token_type_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_mask")

bert = BertLayer(bert_config, is_training=True)

output = bert(input_ids, input_mask=input_mask, token_type_ids=token_type_ids, sequence_output=False)
output = tf.keras.layers.Dense(100, activation=gelu)(output)
output = tf.keras.layers.Dense(5, activation="softmax")(output)
print(output.shape)
model = tf.keras.Model(inputs=[input_ids, input_mask, token_type_ids], outputs=output)
model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len), (None, max_seq_len)])

print("Loading pre-trained weights")
bert.load_stock_weights(model, init_checkpoint)

print(model.summary())

