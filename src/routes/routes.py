from logzero import logger
from flask import request

from src.services import action_handler


def init_app(app):
    @app.route('/google_home/webhook', methods=['POST'])
    def handle_ga_webhook():
        """handles webhooks from  google assistant actions"""

        logger.info('Received webhook request from Google Assistant')

        body = request.get_json()

        return action_handler.handle_ga_action(body)
