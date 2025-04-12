"""
Module for uploading images, processing them with an ML client, and storing results in MongoDB.
This Flask application handles image uploads,
processes images with an ML client, and retrieves stored images.
"""

import os
import io
import base64
from datetime import datetime

import requests
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from PIL import Image, UnidentifiedImageError
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv

load_dotenv()

# Get configuration values from environment variables
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DBNAME = os.getenv("MONGO_DBNAME")
ML_CLIENT_URL = os.getenv("ML_CLIENT_URL")  # URL for the ML picture processing client
MAX_IMAGE_SIZE = 16 * 1024 * 1024  # 16MB in bytes

# Connect to the MongoDB database and use the 'images' collection to store image data
client = MongoClient(MONGO_URI)
db = client[MONGO_DBNAME]
images_collection = db.images


def load_image_from_request():
    """
    Processes the incoming request to load an image either from the uploaded file field
    or from a base64-encoded captured image.
    Returns a tuple (image, filename) or (None, None) on error.
    """
    file_obj = request.files.get("image")
    captured = request.form.get("captured_image", "")

    if file_obj and file_obj.filename:
        try:
            image = Image.open(file_obj)
        except (UnidentifiedImageError, OSError):
            flash("Invalid image format!")
            return None, None
        return image, file_obj.filename

    if captured:
        try:
            parts = captured.split(",", 1)
            if len(parts) != 2:
                flash("Invalid captured image data!")
                return None, None
            encoded = parts[1]
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data))
        except (UnidentifiedImageError, OSError, ValueError):
            flash("Invalid captured image format!")
            return None, None
        return image, "captured.jpg"

    flash("No file selected!")
    return None, None


def process_upload(image_obj, filename):
    """
    Converts the image to JPEG, checks its size, sends it to the ML client,
    and stores the image and prediction in MongoDB.
    Returns the inserted document's ID as a string.
    """
    if image_obj.mode in ("RGBA", "LA"):
        image_obj = image_obj.convert("RGB")

    image_bytes = io.BytesIO()
    image_obj.save(image_bytes, format="JPEG")
    img_data = image_bytes.getvalue()

    if len(img_data) > MAX_IMAGE_SIZE:
        flash("Uploaded image exceeds 16MB and cannot be stored!")
        return None

    image_b64 = base64.b64encode(img_data).decode("utf-8")
    payload = {"image": f"data:image/jpeg;base64,{image_b64}"}

    try:
        ml_response = requests.post(ML_CLIENT_URL, json=payload, timeout=30)
        ml_response.raise_for_status()
        prediction = ml_response.json().get("results", "No result")
    except requests.RequestException as req_err:
        prediction = f"Error during prediction: {req_err}"
    except ValueError:
        prediction = "Error decoding ML response"

    print(prediction, flush=True)

    result = images_collection.insert_one(
        {
            "filename": filename,
            "data": img_data,
            "content_type": "image/jpeg",
            "upload_date": datetime.utcnow(),
            "prediction": prediction,
        }
    )
    return str(result.inserted_id)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handles image upload and displays uploaded images.
    On a POST request:
      - Processes the uploaded image.
      - Checks if its size is <= 16MB.
      - Converts it to JPEG format.
      - Sends it to the ML client for processing.
      - Stores the image and prediction result in MongoDB.
    On a GET request, retrieves and renders all stored images.
    """
    if request.method == "POST":
        image_obj, filename = load_image_from_request()
        if image_obj is None:
            return redirect(request.url)
        new_id = process_upload(image_obj, filename)
        if new_id is None:
            return redirect(request.url)

        flash("Image uploaded and processed successfully!")
        return redirect(url_for("index", uploaded=new_id))

    # For GET requests: retrieve only the newly uploaded document if provided.
    uploaded_id = request.args.get("uploaded")
    if uploaded_id:
        try:
            file_doc = images_collection.find_one({"_id": ObjectId(uploaded_id)})
            files = [file_doc] if file_doc is not None else []
        except (InvalidId, PyMongoError) as err:
            flash(f"Error retrieving image: {err}")
            files = []
    else:
        files = []

    return render_template("index.html", files=files)


@app.route("/uploads/<image_id>")
def get_image(image_id):
    """
    Retrieves an image from the MongoDB collection by its document ID
    and returns it as a file.
    """
    try:
        image_doc = images_collection.find_one({"_id": ObjectId(image_id)})
    except (InvalidId, PyMongoError) as err:
        flash(f"Error retrieving image: {err}")
        return redirect(url_for("index"))

    if image_doc is None:
        flash("Image not found!")
        return redirect(url_for("index"))

    try:
        return send_file(
            io.BytesIO(image_doc["data"]),
            mimetype=image_doc.get("content_type", "image/jpeg"),
            as_attachment=False,
            download_name=image_doc.get("filename", "image.jpg"),
        )
    except Exception as send_err:  # pylint: disable=broad-exception-caught
        flash(f"Error sending image: {send_err}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
