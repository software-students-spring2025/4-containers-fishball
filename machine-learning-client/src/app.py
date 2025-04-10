"""
This module provides a Flask application for analyzing images for faces
and retrieving analysis results from a database.
"""

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


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Endpoint to analyze an uploaded image for faces.
    Returns the analysis results or an error message.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return jsonify({"error": "Invalid file type"}), 400

    temp_path = f"/tmp/{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    file.save(temp_path)

    try:
        results = analyzer.analyze(temp_path)
        if not results:
            return jsonify({"error": "No faces detected"}), 400

        analysis_id = database.store_analysis(temp_path, results)
        os.remove(temp_path)
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

    except (OSError, ValueError) as e:
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:  # Fallback for unexpected runtime errors
        os.remove(temp_path)
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


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
    app.run(host="0.0.0.0", port=5000)
