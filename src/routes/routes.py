from logzero import logger
from flask import request, jsonify

from src.constants.constants import http_responses, general
from src.models import user
from src.services import bot_actions
from src.services import bot_utterances


def init_app(app):
    @app.route("/test")
    def test_fn():
        try:
            users = user.get_uses()

            for db_user in users:
                print(db_user.get('name', 'no name'))
                print(db_user.get('email', 'no email'))

            logger.info(http_responses['SUCCESS_MSG'])
            action = http_responses['SUCCESS_MSG']
        except Exception as e:
            logger.error(http_responses['FAIL_MSG'])
            logger.error(e)
            action = http_responses['FAIL_MSG']
        return action

    @app.route('/webhook', methods=['POST'])
    def handle_webhook():
        """ performs actions triggered from Bot brain (rasa-core server)"""

        logger.info('Received  perform action request')

        try:
            data = request.data
            response = {}
            events = bot_actions.bot_perform_action(data)

            response['events'] = []
            response['responses'] = []

            return jsonify(response)
        except Exception as err:
            logger.error(err)
            return 'Error performing action', 400

    @app.route('/nlg', methods=['POST'])
    def handle_get_utter():
        """ generates utterances for Bot brain (rasa-core server)"""

        try:
            body = request.get_json()

            sender_id = body['tracker'].get('sender_id', general['default_user'])
            input_channel = body['tracker'].get('latest_input_channel', 'Error - no input channel defined')
            slots = body['tracker'].get('slots', [])
            template = body.get('template', 'Error - no template defined')

            # TODO: get user data from db ----------->>>>>>>>>>

            logger.info('Received generate utterance request')
            logger.info(template)
            logger.info(slots)
            logger.info(body['tracker']['latest_message'].get('intent', 'Error - no latest message defined'))
            logger.info(body['tracker']['latest_message'].get('text', 'Error - no latest message text'))
            logger.info(input_channel)

            # prepare data to generate utterance
            data = {
                'sender_id': sender_id,
                'template': template,
                'slots': slots,
                'user_name': '',
                'input_channel': input_channel
            }

            text = bot_utterances.generate_utterance(data)
            buttons = []

            response = {
                "text": text,
                "buttons": buttons,
                "image": None,
                "elements": [],
                "attachments": []
            }

            return jsonify(response)
        except Exception as err:
            logger.error(err)

            return 'Error performing utterance', 400
