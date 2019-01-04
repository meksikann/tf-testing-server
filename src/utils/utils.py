from os.path import join, dirname
from dotenv import load_dotenv
import argparse


def get_env_args():
    # Create .env file path.
    dotenv_path = join(dirname(__file__), '.env')

    # Load file from the path.
    load_dotenv(dotenv_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", action="store", default="8282")
    parser.add_argument("--debug",  action="store", default=False)

    return parser.parse_args()
