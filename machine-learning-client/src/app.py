"""
This module provides a Flask application for analyzing images for faces
and retrieving analysis results from a database.
"""

import base64
import os
import uuid
from flask import Flask, request, jsonify
from face_analyzer import FaceAnalyzer
from db_handler import DBHandler
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

analyzer = FaceAnalyzer()
database = DBHandler()


@app.route("/", methods=["POST"])
def analyze():
    """
    Endpoint to analyze an uploaded image for faces.
    Returns the analysis results or an error message.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Check if image data exists
    if "image" not in data:
        return jsonify({"error": "No file provided"}), 400

    image_data = data["image"]

    try:
        # Extract the actual base64 string
        header, base64_str = image_data.split(",", 1)

        # Determine file extension from header
        if "jpeg" in header:
            ext = ".jpg"
        elif "png" in header:
            ext = ".png"
        else:
            return jsonify({"error": "Unsupported image type"}), 400

        # Decode and save temporarily
        temp_path = f"/tmp/{uuid.uuid4()}{ext}"
        with open(temp_path, "wb") as f:
            f.write(base64.b64decode(base64_str))

        # Analyze the image
        results = analyzer.analyze(temp_path)
        if not results:
            os.remove(temp_path)
            return jsonify({"error": "No faces detected"}), 400

        # Store results
        analysis_id = database.store_analysis(temp_path, results)
        # os.remove(temp_path)

        return (
            jsonify(
                {
                    "analysis_id": str(analysis_id),
                    "results": results,
                    "models": Config.DEEPFACE_MODELS,
                }
            ),
            200,
        )

    except base64.binascii.Error:
        return jsonify({"error": "Invalid base64 encoding"}), 400


@app.route("/analysis/<analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    """
    Endpoint to retrieve analysis results by analysis ID.
    Returns the analysis data or an error message.
    """
    analysis = database.get_analysis(analysis_id)
    if not analysis:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify(analysis)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
