from logzero import logger
import pickle
import json
import numpy as np

from src.brain_processor import helper


def predict_intent(utterance):
    """preprocess text to format which model uses to predict"""
    logger.info('Got utterance to predict')

    formatted_result = 'NONE'

    try:
        words, intents, intents_patterns, model = get_training_data()

        # todo: make words embedding, model prediction and so on ------------------>>>>>>
        vectorized_utterance = vectorize_sentence(utterance, words)

        print(vectorized_utterance)

        return formatted_result
    except Exception as err:
        logger.error(err)
        return formatted_result


def preprocess_sentence(sentence):
    # tokinize words
    words = helper.get_tokenized_words(sentence)

    # steam words
    words = helper.stem_data(words)

    return words


def vectorize_sentence(sentence, words):
    # create np array - bag of words
    sentence_words = preprocess_sentence(sentence)

    # make bag
    bag = [0]*len(words)

    for s_word in sentence_words:
        for idx, word in enumerate(words):
            if word == s_word:
                bag[idx] = 1

    return(np.array(bag))



def get_training_data():
    """get data saved during training process"""
    intents_patterns_path, model_dir, training_data_dir = helper.get_training_data_dirs()
    try:
        # load training data to process words
        data = pickle.load(open(training_data_dir, 'rb'))
        words = data['words']
        intents = data['intents']
        x_train = data['x_train']
        y_train = data['y_train']

        intents_patterns = json.loads(open(intents_patterns_path).read())

        # define same model, which used for training
        model = helper.get_dnn_model(x_train, y_train)

        model.load(model_dir)

        return words, intents, intents_patterns, model
    except Exception as err:
        logger.error(err)
        raise Exception('Error while getting training data')
