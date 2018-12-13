import os
from pymongo import MongoClient
from src.config import config

client = MongoClient(config.MONGO_URL, config.MONGO_PORT)
db = client['cpd-bot']
# secret_key = os.getenv('SECRET_KEY')
# print('SECRET KEY ', secret_key)

def get_uses():
    users_collection = db.users
    print(config.MONGO_PORT)
    users = users_collection.find()
    return users


