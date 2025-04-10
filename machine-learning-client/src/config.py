"""
Configuration module for the machine learning client.
Loads environment variables and provides configuration settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Configuration class for managing application settings.
    """

    # DeepFace
    DEEPFACE_BACKEND = os.getenv("DEEPFACE_BACKEND", "opencv")
    DEEPFACE_MODELS = os.getenv("DEEPFACE_MODELS", "age,gender,emotion").split(",")
    DETECTOR_THRESHOLD = float(os.getenv("DETECTOR_THRESHOLD", "0.9"))
    ENFORCE_DETECTION = os.getenv("ENFORCE_DETECTION", "true").lower() == "true"

    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB = os.getenv("MONGO_DB")

    # Flask
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    API_KEY = os.getenv("API_KEY")
    DEBUG = os.getenv("DEBUG_MODE", "false").lower() == "true"

    @staticmethod
    def is_debug_mode():
        """
        Check if the application is running in debug mode.
        """
        return Config.DEBUG

    @staticmethod
    def get_mongo_uri():
        """
        Retrieve the MongoDB connection URI.
        """
        return Config.MONGO_URI
