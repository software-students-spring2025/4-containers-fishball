"""Machine learning client that detects emotions using FER and MongoDB."""

import io
import os
import time

from dotenv import load_dotenv
from pymongo import MongoClient
from PIL import Image, UnidentifiedImageError
from fer import FER
import gridfs
from gridfs.errors import NoFile

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DBNAME = os.getenv("MONGO_DBNAME")
COLLECTION = "images"

client = MongoClient(MONGO_URI)
db = client[MONGO_DBNAME]
fs = gridfs.GridFS(db)
collection = db[COLLECTION]
analyzer = FER()

def analyze_image(image_bytes):
    """Run FER analysis on an image and return detected emotions."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = analyzer.detect_emotions(image)
    return result

def process_pending_images():
    """Find unprocessed images in the DB, analyze, and update."""
    pending = collection.find({"status": "pending"})
    for doc in pending:
        file_id = doc.get("file_id")
        if not file_id:
            print(f"Skipping invalid document: {doc}")
            continue

        print(f"Processing image with file_id: {file_id}")
        try:
            image_bytes = fs.get(file_id).read()
            result = analyze_image(image_bytes)
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"status": "complete", "result": result}}
            )
            print(f"Analysis complete for {file_id}")
        except (NoFile, UnidentifiedImageError) as e:
            print(f"Known processing error: {e}")

if __name__ == "__main__":
    print("Machine Learning client started. Waiting for new images...")
    while True:
        process_pending_images()
        time.sleep(5)
