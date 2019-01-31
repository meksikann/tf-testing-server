from logzero import logger
import pickle
import json
import numpy as np
import random
import os
import traceback

from src.brain_processor import helper


def predict_intent(utterance):
    """preprocess text to format which model uses to predict"""
    logger.info('Got utterance to predict')

    formatted_result = 'NONE'

    model_env = os.environ.get("model_env")

    try:
        if model_env == 'tf':
            print('Using TF model for prediction --------->>>>>>>')
            tokenized_utterance = helper.get_tokenized_utterance(utterance)
            print('Predict X length:', len(tokenized_utterance))
            model = helper.get_tf_model(len(tokenized_utterance))
            model.load_weights('nlu_mlp_model.h5')
            print('data to predict: ', tokenized_utterance)
            tokenized_utterance = (np.expand_dims(tokenized_utterance, 0))
            prediction = model.predict(tokenized_utterance, verbose=1)
            classes = helper.get_tf_classes()

            print('classes ', classes)
            print('predicted: ', prediction)
            print('shape', prediction.shape)

            for pred in prediction:
                score = np.argmax(pred)
                print(f'Index: {score}, Class is: {classes[score]}')
        else:
            words, intents, intents_patterns, model = get_training_data()
            vectorized_utterance = vectorize_sentence(utterance, words)
            predictions = classify_vector(vectorized_utterance, model, intents)

            print('predictions ', predictions)

            classes = intents_patterns['intents']

            # print(classes)

            if predictions and len(predictions) > 0:
                for cl in classes:
                    # compare with first prediction
                    if cl['tag'] == predictions[0][0]:
                        formatted_result = random.choice(cl['responses'])

        return formatted_result
    except Exception as err:
        logger.error(err)
        traceback.print_exc()
        return formatted_result


def classify_vector(vector, model, intents):
    NLU_TRECHHOLD = 0.25
    print('vector', vector)

    predictions = model.predict([vector])[0]

    logger.info('Predictions: ')
    logger.info(predictions)
    # filter out predictions lower than threshold

    predictions = [[i, pred] for i, pred in enumerate(predictions) if pred > NLU_TRECHHOLD]

    predictions.sort(key=lambda x: x[1], reverse=True)

    predicted_list = []

    for p in predictions:
        predicted_list.append((intents[p[0]], p[1]))
    return predicted_list


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
    bag = [0] * len(words)

    for s_word in sentence_words:
        for idx, word in enumerate(words):
            if word == s_word:
                bag[idx] = 1

    return np.array(bag)


def get_training_data():
    """get data saved during training process"""
    intents_patterns_path, model_dir, training_data_dir, tf_training_data_dir, model_weights_dir = helper.get_training_data_dirs()
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
