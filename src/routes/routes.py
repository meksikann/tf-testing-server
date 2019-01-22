from logzero import logger
from flask import request

from src.services import action_handler
from src.brain_processor import training
from src.brain_processor import predictor


def init_app(app):
    @app.route('/google_home/webhook', methods=['POST'])
    def handle_ga_webhook():
        """handles webhooks from  google assistant actions"""

        logger.info('Received webhook request from Google Assistant')

        body = request.get_json()

        return action_handler.handle_ga_action(body)

    @app.route('/start-nlu-training', methods=['POST'])
    def start_nlu_training():
        """made to trigger nlp training"""

        logger.info('Start NLP training ...')
        result = training.start_nlu_training()

        return result or 'Training finished'

    @app.route('/predict', methods=['POST'])
    def start_nlp_prediction():
        logger.info('Start nlp prediction ....')
        body = request.get_json()
        print('utterance: ', body['utterance'])
        utterance = body['utterance']

        result = predictor.predict_intent(utterance)
        return result
