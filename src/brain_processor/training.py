import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import numpy as np
import tensorflow as tf
import random
import json
from os.path import join, dirname

nltk.download('punkt')
print('TF version:' ,tf.__version__)

# tf.enable_eager_execution()


def start_nlu_training():
    # load defined intents from json file
    intents_patterns_path = join(dirname(__file__), 'intents.json')
    model_dir = join(dirname(__file__), '../trained_models/model_1.tflearn')
    intents_patterns = json.loads(open(intents_patterns_path).read())

    # generate training data with format needed to train model
    x_train, y_train = generate_training_data(intents_patterns)

    print('training data', x_train[0])

    model = get_dnn_model(x_train, y_train)

    model.fit(x_train, y_train, n_epoch=1000, batch_size=8, show_metric=True)

    model.save(model_dir)

    print('MODEL HAS BEEN TRAINED --------------------------->>>>>>>>>>>>>>>>>>')


# todo: ------------>>>>
def get_dnn_model(x_train, y_train):
    # reset graph data
    tf.reset_default_graph()

    # build NN
    net = tflearn.input_data(shape=[None, len(x_train[0])])
    net = tflearn.fully_connected(net, 4)
    net = tflearn.fully_connected(net, 4)
    net = tflearn.fully_connected(net, len(y_train[0]), activation='softmax')

    net = tflearn.regression(net,  optimizer='adam')

    # define model
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

    return model



def generate_training_data(intents_patterns):
    # create stemmer for sequences normalization
    stemmer = LancasterStemmer()

    documents, words, intents = generate_stemmed_data(intents_patterns, stemmer)

    training_set = generate_training_set(documents, words, intents, stemmer, len(intents))

    # print('Training set', training_set)

    # shuffle training set
    random.shuffle(training_set)

    # turn list into numpy array
    training_set = np.array(training_set)

    # create training X and Y lists
    x_train = list(training_set[:, 0])
    y_train = list(training_set[:, 1])

    return x_train, y_train


def generate_training_set(documents, words, intents, stemmer, intents_amount):
    # receives set of documents (sentense and intent asociated with it) and transforms it to tensors fo numbers
    # returns training set X, Y
    # #

    training = []
    output = []
    out_empty = [0] * intents_amount

    # iterate documents - generate bag of words for each sentence
    for doc in documents:
        bag = []
        tokenized_words = doc[0]

        # stemm intent related sentence
        tokenized_words = [stemmer.stem(word.lower()) for word in tokenized_words]

        # push words to bag
        # generate X - make vector  - vectors should be same length , so used words array to build vector with 1-match 0-not match words
        for word in words:
            bag.append(1) if word in tokenized_words else bag.append(0)

        # generate Y
        y_row = list(out_empty)
        y_row[intents.index(doc[1])] = 1

        training.append([bag, y_row])

    return training


def generate_stemmed_data(intents_patterns, stemmer):
    words = []
    intents = []
    skip_words = ['?']
    documents = []

    # prepare data (tokenize words)
    for intent in intents_patterns['intents']:
        #  tokenize each word in patterns
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            print(f'tokenized word {pattern} into token: {word}')

            # add words to list - need for further vectors building
            words.extend(word)

            # add word to documents - list of sentences and intents associated with it
            documents.append((word, intent['tag']))
            # print(documents)

            # add to intents (classes)
            if intent['tag'] not in intents:
                intents.append(intent['tag'])

        # make to words lower case and stem all words which are available
        words = [stemmer.stem(word.lower()) for word in words if word not in skip_words]

        # remove duplicates using set() method
        words = sorted(list(set(words)))

    return documents, words, intents
