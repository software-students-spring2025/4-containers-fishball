"""
This module provides a Flask application for analyzing images for faces
and retrieving analysis results from a database.
"""

import io
import base64
import os
import uuid
from pymongo import MongoClient
from flask import Flask, request, jsonify, redirect, send_file, flash
from face_analyzer import FaceAnalyzer
from db_handler import DBHandler
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY or "test_secret"
app.config["TESTING"] = True

analyzer = FaceAnalyzer()
database = DBHandler()
images_collection = MongoClient(Config.MONGO_URI)[Config.MONGO_DBNAME]["images"]
app.images_collection = images_collection


def error_response(message, status_code):
    """Helper function to generate an error JSON response."""
    return jsonify({"error": message}), status_code


@app.route("/", methods=["POST"])
def analyze():
    """
    Endpoint to analyze an uploaded image for faces.
    Returns a JSON response for JSON requests and a redirect for form-data uploads.
    """
    if request.is_json:
        data = request.get_json()
        # Check explicitly if "image" key is missing.
        if "image" not in data:
            return error_response("No file provided", 400)

        try:
            image_data = data["image"]
            header, base64_str = image_data.split(",", 1)
            if "jpeg" in header:
                ext = ".jpg"
            elif "png" in header:
                ext = ".png"
            else:
                raise ValueError("Unsupported image type")
            temp_path = f"/tmp/{uuid.uuid4()}{ext}"
            with open(temp_path, "wb") as f:
                f.write(base64.b64decode(base64_str))
            results = analyzer.analyze(temp_path)
            if not results:
                os.remove(temp_path)
                raise ValueError("No faces detected")
            analysis_id = database.store_analysis(temp_path, results)
            response = jsonify(
                {
                    "analysis_id": analysis_id,
                    "results": results,
                    "models": Config.DEEPFACE_MODELS,
                }
            )
            return response, 200
        except (ValueError, base64.binascii.Error) as e:
            return error_response(str(e), 400)

    # Process form-data file uploads.
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("No file provided")
        return redirect(request.url)

    ext = os.path.splitext(file.filename)[1] or ".jpg"
    temp_path = f"/tmp/{uuid.uuid4()}{ext}"
    file.save(temp_path)

    results = analyzer.analyze(temp_path)
    if not results:
        os.remove(temp_path)
        flash("No faces detected")
        return redirect(request.url)

    analysis_id = database.store_analysis(temp_path, results)
    return redirect(f"/uploads/{analysis_id}")


@app.route("/uploads/<analysis_id>", methods=["GET"])
def get_image(analysis_id):
    """
    Retrieves and sends the image file corresponding to the given analysis_id.
    If the image is not found, flashes an error and redirects to the root URL.
    """
    doc = app.images_collection.find_one({"analysis_id": analysis_id})
    if not doc:
        flash("Image not found")
        return redirect(request.url_root)
    return send_file(
        io.BytesIO(doc["data"]),
        mimetype=doc["content_type"],
        as_attachment=True,
        download_name=doc["filename"],
    )


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
