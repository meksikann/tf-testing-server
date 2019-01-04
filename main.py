#!/usr/bin/env python3
from flask import Flask
from flask_cors import CORS
from logzero import logger

from src.routes import routes
from src.utils import utils

def create_app(config=None):
    app = Flask(__name__)

    # See http://flask.pocoo.org/docs/latest/config/
    # app.config.update()
    app.config.update(config or {})

    # Setup cors headers to allow all domains
    # https://flask-cors.readthedocs.io/en/latest/
    CORS(app)

    # init routes
    routes.init_app(app)
    return app

if __name__ == "__main__":
    logger.info('Starting Jarvis Robot....')

    args = utils.get_env_args()
    port = int(args.port)
    debug = bool(args.debug)
    init_configs = dict(DEBUG=debug)
    app = create_app(init_configs)

    app.run(host="0.0.0.0", port=port)
