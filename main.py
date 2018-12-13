#!/usr/bin/env python3
from os.path import join, dirname
from dotenv import load_dotenv
import argparse
from flask import Flask
from flask_cors import CORS
from src.routes import routes
from logzero import logger


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
    # Create .env file path.
    dotenv_path = join(dirname(__file__), '.env')

    # Load file from the path.
    load_dotenv(dotenv_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", action="store", default="8282")
    parser.add_argument("--debug",  action="store", default=False)

    args = parser.parse_args()
    port = int(args.port)
    debug = bool(args.debug)
    init_configs = dict(DEBUG=debug)
    app = create_app(init_configs)
    app.run(host="0.0.0.0", port=port)
