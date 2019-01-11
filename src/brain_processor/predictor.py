from logzero import logger
import pickle
import json

from src.brain_processor import helper


def predict_intent(utterance):
    """preprocess text to format which model uses to predict"""
    logger.info('Got utterance to predict')

    formatted_result = 'NONE'

    try:
        words, intents, intents_patterns, model = get_training_data()

        # todo: make words embedding, model prediction and so on ------------------>>>>>>

        return formatted_result
    except Exception as err:
        logger.error(err)


def get_training_data():
    """get data saved during training process"""
    intents_patterns_path, model_dir, training_data_dir = helper.get_training_data_dirs()
    try:
        # load training data to process words
        data = pickle.load(training_data_dir, 'rb')
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
