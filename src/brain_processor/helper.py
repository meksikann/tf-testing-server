import tflearn
import tensorflow as tf
from os.path import join, dirname
import nltk
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')

print('TF version:', tf.__version__)


# tf.enable_eager_execution()

def get_tokenized_words(sentence):
    return nltk.word_tokenize(sentence)


def stem_data(words):
     stemmer = LancasterStemmer()

     stemmed_words = [stemmer.stem(word.lower()) for word in words]

     return stemmed_words


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

    return intents_patterns_path, model_dir, training_data_dir
