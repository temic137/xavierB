# import os

# class Config:
#     SQLALCHEMY_DATABASE_URI = 'sqlite:///crm.db'
#     SQLALCHEMY_TRACK_MODIFICATIONS = False
#     SECRET_KEY = 'your_secret_key_here'
#     SESSION_TYPE = 'filesystem'
#     UPLOAD_FOLDER = 'uploads'


import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///crm.db')  # Use SQLite for local dev
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'spacex42695'
    SESSION_TYPE = 'filesystem'
    UPLOAD_FOLDER = 'uploads'

# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Config:
#     SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
#     if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
#         SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)
#     SQLALCHEMY_TRACK_MODIFICATIONS = False
#     SECRET_KEY = 'your_secret_key_here'
#     SESSION_TYPE = 'filesystem'
#     UPLOAD_FOLDER = 'uploads'
