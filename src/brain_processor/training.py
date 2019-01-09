import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import random
import json
from os.path import join, dirname
nltk.download('punkt')

tf.enable_eager_execution()


def start_nlu_training():
    # create stemmer for sequences normalization
    stemmer = LancasterStemmer()

    words = []
    intents = []
    skip_words = ['?']
    documents = []

    # load defined intents from json file
    intents_patterns_path = join(dirname(__file__), 'intents.json')
    intents_patterns = json.loads(open(intents_patterns_path).read())
    print(intents_patterns)

    # prepare data (tokenize words)
    for intent in intents_patterns['intents']:
        #  tokenize each word in patterns
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            print(f'tokenized word {pattern} into token: {word}')

            # add words to list TODO: WTF list ???????????????????
            words.extend(word)

            # add word to documents TODO: WTF documents are for ?????????????
            documents.append((word, intent['tag']))
            #print(documents)

            # add to intents (classes)
            if intent['tag'] not in intents:
                intents.append(intent['tag'])

        # make to words lower case and stem each word
        words = [stemmer.stem(word.lower()) for word in words if word not in skip_words]

        # remove duplicates using set() method
        words = sorted(list(set(words)))

        print(len(documents), 'documents')
        print(len(intents), 'intents', intents)
        print(len(words), 'stemmed words', words)










