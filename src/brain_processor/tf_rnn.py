from __future__ import absolute_import, division, print_function

from os.path import join, dirname
import tensorflow as tf

tf.enable_eager_execution()
import  numpy as np
import os
import time

import helper



print('TF VERSION', tf.__version__)

def generate_input_target(batch):
    x_train = batch[: -1]
    y_train = batch[1:]

    return x_train, y_train

# file_path = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# file_path = '/home/serg/.keras/datasets/shakespeare.txt'


file_path = join(dirname(__file__),'book_dataset_rnn.txt')
print(file_path)

text = open(file_path, 'rb').read().decode(encoding='utf-8')

# get vocab of unique chars /case sensitive/
characters_vocab = sorted(set(text))

# create two hashes  'char->int' and 'int->char'
char_to_idx = {u: i for i, u in enumerate(characters_vocab)}
idx_to_char = np.array(characters_vocab)


# tokenize all text
tokenized_text = np.array([char_to_idx[char] for char in text])

sequence_length = 100
epoch_sequence_length = len(text) // sequence_length

# convert text into stream of characters from tesnor - we feed char by char as input stream
dataset_stream = tf.data.Dataset.from_tensor_slices(tokenized_text)

# use batches to feed sized batches (with fixed characters ) to model
char_streams = dataset_stream.batch(sequence_length, drop_remainder=True)

dataset = char_streams.map(generate_input_target)

# create training batches
BUFFER_SIZE = 10000  # buffer in TF to shuffles data
BATCH_SIZE = 60
steps_per_epoch = epoch_sequence_length // BATCH_SIZE

# shuffle ans split by batches
dataset = dataset.shuffle(BATCH_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# build a sequential model

vocab_size = len(characters_vocab)
embedding_dimension = 256  # TODO: WTF??????
rnn_units = 1024  # TODO: WTF?

model = helper.build_model(
    vocab_size=vocab_size,
    embed_dimention=embedding_dimension,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE
)

model.summary()


# loss function passed
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=loss
)


# setup checkpints
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# fit_history = model.fit(dataset.repeat(), epochs=3, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])


def generate_text(string):
    generate_num = 10000
    generated_text = []
    temperature = 1.0

    model = helper.build_model(vocab_size, embedding_dimension, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    # vectorize input  string
    input_value = [char_to_idx[str] for str in string]
    input_value = tf.expand_dims(input_value, 0)

    # reset model states
    model.reset_states()

    for i in range(generate_num):
        predictions = model(input_value)

        # remove batch dimension
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

        # pass predicted valua as next input
        input_value = tf.expand_dims([predicted_id], 0)
        generated_text.append(idx_to_char[predicted_id])

    return string + ''.join(generated_text)


print(generate_text(string="Neural network models are a preferred method for developing statistical language models because they can use a distributed representation where different words with similar meanings have similar representation and because they can use a large context of recently observed words when making predictions"))




