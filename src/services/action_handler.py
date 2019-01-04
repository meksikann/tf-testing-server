import json
from os.path import join, dirname
from logzero import logger
from flask import jsonify


def handle_ga_action(body):
    ACTIVE_CONVERSATION = 'ACTIVE'
    NEW_CONVERSATION = 'NEW'

    try:
        # check conversation stage (start, active or end)

        if body['conversation']['type'] == NEW_CONVERSATION:
            answer_template_path = join(dirname(__file__), '../answer_templates/start_response.json')
            answer_template = json.loads(open(answer_template_path).read())

        elif body['conversation']['type'] == ACTIVE_CONVERSATION:
            # TODO: make the magic here ===============================>>>>>>>>>>>>>>>>>>

            answer_template_path = join(dirname(__file__), '../answer_templates/casual_response.json')
            answer_template = json.loads(open(answer_template_path).read())

        else:
            print(body['conversation'])
            answer_template_path = join(dirname(__file__), '../answer_templates/stop_response.json')
            answer_template = json.loads(open(answer_template_path).read())

        response = answer_template

        return jsonify(response)
    except Exception as err:
        logger.error(err)
        return 'Error performing ga request'
