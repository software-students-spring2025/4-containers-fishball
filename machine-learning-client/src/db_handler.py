"""
Database handler module for managing MongoDB operations.
"""

import uuid
from datetime import datetime, timezone
from pymongo import MongoClient
from config import Config


class DBHandler:
    """
    A class to handle database operations for storing and retrieving analyses.
    """

    def __init__(self):
        """
        Initializes the database handler with a MongoDB client and database.
        """
        self.client = MongoClient(Config.MONGO_URI)
        self.database = self.client[Config.MONGO_DBNAME]

    def store_analysis(self, image_path, results):
        """
        Stores analysis results in the database.

        Args:
            image_path (str): Path to the analyzed image.
            results (dict): Analysis results.

        Returns:
            ObjectId: The ID of the inserted document.
        """
        doc = {
            "analysis_id": str(uuid.uuid4()),
            "image_path": image_path,
            "results": results,
            "models": Config.DEEPFACE_MODELS,
            "timestamp": datetime.now(timezone.utc),
        }
        return self.database.analyses.insert_one(doc).inserted_id

    def get_analysis(self, analysis_id):
        """
        Retrieves an analysis document from the database.

        Args:
            analysis_id (str): The ID of the analysis to retrieve.

        Returns:
            dict: The analysis document, or None if not found.
        """
        return self.database.analyses.find_one({"analysis_id": analysis_id})
