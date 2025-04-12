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
        file = request.files["image"]
        captured_image = request.form.get("captured_image", "")
        
        # Check if neither a file was uploaded nor a captured image provided
        if (not file or file.filename == "") and not captured_image:
            flash("No file selected!")
            return redirect(request.url)

        if file.filename == "":
            flash("No file selected!")
            return redirect(request.url)

        try:
            # If a file is uploaded via file input, use that.
            if file and file.filename != "":
                im = Image.open(file)
            else:
                # Otherwise, process the captured image (data URL format)
                # captured_image is expected to be like "data:image/jpeg;base64,..."
                header, encoded = captured_image.split(",", 1)
                image_data = base64.b64decode(encoded)
                im = Image.open(io.BytesIO(image_data))
        except (UnidentifiedImageError, OSError):
            flash("Invalid image format!")
            return redirect(request.url)
        
        if im.mode in ("RGBA", "LA"):
            im = im.convert("RGB")

        # Save the image to an in-memory buffer as JPEG
        image_bytes = io.BytesIO()
        im.save(image_bytes, format="JPEG")
        img_data = image_bytes.getvalue()

        # Check if the image size exceeds the 16MB limit
        if len(img_data) > MAX_IMAGE_SIZE:
            flash("Uploaded image exceeds 16MB and cannot be stored!")
            return redirect(request.url)

        # Prepare the image for ML client processing by encoding it in base64.
        # The ML client may expect a data URL format: "data:image/jpeg;base64,...."
        image_b64 = base64.b64encode(img_data).decode("utf-8")
        payload = {"image": f"data:image/jpeg;base64,{image_b64}"}

        try:
            # Send the image data to the ML client
            ml_response = requests.post(ML_CLIENT_URL, json=payload, timeout=30)
            ml_response.raise_for_status()  # Raises an exception if the response is not 200
            prediction = ml_response.json().get("results", "No result")
        except requests.RequestException as req_err:
            prediction = f"Error during prediction: {req_err}"
        except ValueError as val_err:
            prediction = f"Error decoding ML response: {val_err}"

        print(prediction, flush=True)
        # Build the image document for MongoDB, including prediction result
        image_doc = {
            "filename": file.filename,
            "data": img_data,
            "content_type": "image/jpeg",
            "upload_date": datetime.utcnow(),
            "prediction": prediction,
        }

        # Insert the image document into the MongoDB collection
        images_collection.insert_one(image_doc)
        flash("Image uploaded and processed successfully!")
        return redirect(url_for("index"))

    # For GET requests, retrieve all images from the database
    files = list(images_collection.find())
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
