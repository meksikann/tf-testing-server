import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
import pickle
from logzero import logger
import os
import traceback

from src.brain_processor import helper
from src.utils import utils


def start_nlu_training():
    try:
        intents_patterns_path, model_dir, training_data_dir, tf_training_data_dir, model_weights_dir = helper.get_training_data_dirs()
        # load defined intents from json file
        intents_patterns = json.loads(open(intents_patterns_path).read())
        model_env = os.environ.get("model_env")

        if model_env == 'tf':
            x_train, y_train, classes, word_indexes = helper.preproces_tf_training_data(intents_patterns)

            print('x_train: ', x_train)
            print('y_train: ', y_train)
            # print('classes: ', classes)

            print('Use Pure TensorFlow model: ==================>>>>>>>>>>>>>>>>>>>>>')
            model = helper.get_tf_model()

            model.fit(x_train,
                      y_train,
                      epochs=1000,
                      batch_size=10,
                      verbose=1
            )

            # save model weights
            model.save('nlu_mlp_model.h5')
            # save data structures into file (will be used for prediction process)
            pickle.dump({'words_indexes': word_indexes, "classes": classes, 'x_train': x_train, 'y_train': y_train},
                        open(tf_training_data_dir, "wb"))
            logger.info('Training data saved')
        else:
            # generate training data with format needed to train model
            x_train, y_train, words, intents = generate_training_data(intents_patterns)

            model = helper.get_dnn_model(x_train, y_train)

            model.fit(x_train, y_train, n_epoch=1000, batch_size=8, show_metric=True)

            model.save(model_dir)

            # save data structures into file (will be used for prediction process)
            pickle.dump({'words': words, "intents": intents, 'x_train': x_train, 'y_train': y_train},
                        open(training_data_dir, "wb"))

        logger.info('MODEL HAS BEEN TRAINED --------------------------->>>>>>>>>>>>>>>>>>')
    except Exception as err:
        logger.error('Error while model training', err)

        traceback.print_exc()


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
    x_train = training_set[:, 0]
    y_train = training_set[:, 1]

    return x_train, y_train, words, intents


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
            word = helper.get_tokenized_words(pattern)
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
