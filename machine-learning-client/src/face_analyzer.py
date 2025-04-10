"""
This module provides a FaceAnalyzer class for analyzing facial attributes using DeepFace.
"""

import logging
from deepface import DeepFace
from config import Config


class FaceAnalyzer:
    """
    A class to analyze facial attributes using DeepFace.
    """

    def __init__(self):
        """
        Initializes the FaceAnalyzer with configuration settings.
        """
        self.config = Config()

    def analyze(self, image_path):
        """
        Analyzes the given image for facial attributes.

        Args:
            image_path (str): Path to the image to be analyzed.

        Returns:
            dict: Analysis results or None if an error occurs.
        """
        try:
            return DeepFace.analyze(
                img_path=image_path,
                actions=self.config.DEEPFACE_MODELS,
                detector_backend=self.config.DEEPFACE_BACKEND,
                enforce_detection=self.config.ENFORCE_DETECTION,
                silent=True,
            )
        except ValueError as e:
            logging.error("Analysis failed: %s", str(e))
            return None
