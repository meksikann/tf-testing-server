import tflearn
import tensorflow as tf
import random
import nltk
import numpy as np
import pickle

from os.path import join, dirname
from nltk.stem.lancaster import LancasterStemmer
from tensorflow import keras
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

nltk.download('punkt')


# tf.enable_eager_execution()
MAX_LENGTH = 10
NORMALIZE_NUM = 50


def parse_training_data(data):
    """receives  JSON format and parses to array with sentences and labels array"""
    training_data = []
    classes = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_data.append([pattern, intent['tag']])
        # generate classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    return training_data, classes


def preprocess_labels(labels, classes):
    y_train = []

    for label in labels:
        y_train.append([classes.index(label)])

    one_hot_labels = keras.utils.to_categorical(np.array(y_train), num_classes=len(classes))

    return one_hot_labels


def preproces_tf_training_data(data):
    """receives JSON  data for training. Returns x_train. y_train vectorized data and classes(intents)"""

    # max features for tokenizer
    MAX_FEATURES = 10000
    # max sequence length
    MAX_LENGTH = 200

    # get training data from JSON, shuffle it and make numpy array
    training_data, classes = parse_training_data(data)

    # random.shuffle(training_data)
    training_data = np.array(training_data)

    # init tokenizer
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)

    # vocabluary with training texts (tokenized) and vectorize training text
    texts = training_data[:, 0]
    tokenizer.fit_on_texts(texts)

    x_train = tokenizer.texts_to_sequences(texts)

    # get max sequence length
    max_length = len(max(x_train, key=len))

    if max_length > MAX_LENGTH:
        max_length = MAX_LENGTH

    # pad sequences to same length
    x_train = sequence.pad_sequences(x_train, 10)
    x_train = x_train / NORMALIZE_NUM

    # generate labels as numbers (classes)
    y_train = preprocess_labels(training_data[:, 1], classes)

    return x_train, y_train, classes, tokenizer.word_index



def get_tokenized_utterance(utterance):
    tokenizer = text.Tokenizer()

    tokenizer.word_index = get_tokens_dict()

    print('word index',tokenizer.word_index)

    x_train = tokenizer.texts_to_sequences([utterance])
    x_train = sequence.pad_sequences(x_train, 10)
    print('before normalization', x_train)
    x_train = x_train / NORMALIZE_NUM

    return x_train


def get_tokens_dict():
    tf_training_data_dir = join(dirname(__file__), 'tf_training_data')
    data = pickle.load(open(tf_training_data_dir, 'rb'))
    word_index = data['words_indexes']

    return word_index


def get_tf_classes():
    tf_training_data_dir = join(dirname(__file__), 'tf_training_data')
    data = pickle.load(open(tf_training_data_dir, 'rb'))
    classes = data['classes']

    return classes


def get_tokenized_words(sentence):
    return nltk.word_tokenize(sentence)


def stem_data(words):
    stemmer = LancasterStemmer()
    stemmed_words = [stemmer.stem(word.lower()) for word in words]

    return stemmed_words

def get_tf_model(length):
    tf.reset_default_graph()
    print('TF version:', tf.__version__)

    model = keras.Sequential()
    model.add(keras.layers.Dense(32, activation=tf.nn.relu, input_dim=10))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(3, activation=tf.nn.softmax))


    model.summary()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_dnn_model(x_train, y_train):
    # reset graph data
    tf.reset_default_graph()

    # build NN
    net = tflearn.input_data(shape=[None, len(x_train[0])])
    net = tflearn.fully_connected(net, 4)
    net = tflearn.fully_connected(net, 4)
    net = tflearn.fully_connected(net, len(y_train[0]), activation='softmax')

    net = tflearn.regression(net, optimizer='adam')

    # define model
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    return model


def get_training_data_dirs():
    intents_patterns_path = join(dirname(__file__), 'intents.json')
    model_dir = join(dirname(__file__), '../trained_models/model_1.tflearn')
    training_data_dir = join(dirname(__file__), 'training_data')
    tf_training_data_dir = join(dirname(__file__), 'tf_training_data')
    model_weights_dir = join(dirname(__file__), 'weights/tf_weights')

    return intents_patterns_path, model_dir, training_data_dir, tf_training_data_dir, model_weights_dir
