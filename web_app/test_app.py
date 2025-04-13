"""
Test module for src.app endpoints and functionality.
"""

import io
import os
import pytest
from PIL import Image
from bson.objectid import ObjectId
from requests import RequestException
from werkzeug.datastructures import FileStorage
import base64
from src.app import (
    load_image_from_file,
    load_image_from_capture,
    process_upload,
    app,
)

os.environ.setdefault("SECRET_KEY", "test_secret_key")
os.environ.setdefault("MONGO_DBNAME", "test_db")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/test_db")
os.environ.setdefault("ML_CLIENT_URL", "http://fake-ml-client")


def generate_valid_objectid():
    """Generate and return a valid ObjectId string."""
    return str(ObjectId())


# pylint: disable=unused-argument
class FakeImagesCollection:
    """A fake collection to simulate MongoDB image operations in tests."""

    def __init__(self):
        self.data = {}

    def insert_one(self, doc):
        """Simulate a document insert by assigning a fixed test_id."""
        doc["_id"] = "test_id"
        self.data["test_id"] = doc

        # pylint: disable=too-few-public-methods
        class DummyResult:
            """A dummy result object that simulates the insert_one return value."""

            inserted_id = "test_id"

        return DummyResult()

    def find(self):
        """Return all stored documents as a list."""
        return list(self.data.values())

    def find_one(self, query):
        """Return a single document based on the _id in the query."""
        _id = query.get("_id")
        return self.data.get(_id, None)


@pytest.fixture(autouse=True)
def fake_images_collection(monkeypatch):
    """Fixture to override the images_collection with a fake collection."""
    fake_collection = FakeImagesCollection()
    monkeypatch.setattr("src.app.images_collection", fake_collection)
    return fake_collection


class FakeResponse:
    """A fake response object to simulate the return value of requests.post."""

    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        """Raise a ValueError if the status_code indicates an error."""
        if self.status_code != 200:
            raise ValueError("Bad response")

    def json(self):
        """Return the stored JSON data."""
        return self._json


@pytest.fixture
def fake_requests_post(monkeypatch):
    """
    Fixture to override requests.post used in src.app with a fake function.

    The fake_post function ignores its arguments and returns a FakeResponse.
    """

    def fake_post(_url, _json, _timeout):
        return FakeResponse({"result": "happy"})

    monkeypatch.setattr("src.app.requests.post", fake_post)


def test_index_get():
    """Test that the GET '/' endpoint returns 200 OK and contains upload text."""
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert b"Upload" in response.data or b"upload" in response.data


def test_index_post_no_file():
    """Test POST without a file returns error flash and redirect."""
    with app.test_client() as client:
        response = client.post("/upload", data={}, follow_redirects=True)
        assert response.status_code == 200
        assert b"No file selected" in response.data


def test_index_post_empty_filename():
    """Test POST with an empty file name returns a 'No file selected' error."""
    with app.test_client() as client:
        data = {"image": (io.BytesIO(b"dummy data"), "")}
        response = client.post("/upload", data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"No file selected" in response.data


def test_index_post_invalid_image():
    """
    Test POST with an invalid image simulating a PIL error.

    The monkeypatch argument has been removed since it was unused.
    """
    fake_file = io.BytesIO(b"not an image")
    fake_file.name = "invalid.jpg"
    with app.test_client() as client:
        data = {"image": (fake_file, "invalid.jpg")}
        response = client.post("/upload", data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Invalid image format!" in response.data


def test_index_post_success():
    """Test successful POST: a valid image is processed and stored."""
    img = Image.new("RGB", (10, 10), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    with app.test_client() as client:
        data = {"image": (img_bytes, "test.jpg")}
        response = client.post(
            "/upload",
            data=data,
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Image uploaded and processed successfully!" in response.data


def test_get_image_not_found():
    """
    Test that a GET request to a non-existent image ID redirects to the index.

    The non-existent ID is a placeholder value.
    """
    non_existent_id = "000000000000000000000000"
    with app.test_client() as client:
        response = client.get(f"/uploads/{non_existent_id}", follow_redirects=True)
        assert response.status_code == 200
        assert b"File Upload" in response.data


def test_load_image_from_file_invalid(monkeypatch):
    fake_file = io.BytesIO(b"not an image")
    fake_file.filename = "invalid.jpg"

    with app.test_request_context(method="POST", data={"image": fake_file}):
        image, filename = load_image_from_file()
        assert image is None
        assert filename is None


def test_load_image_from_capture_invalid(monkeypatch):
    with app.test_request_context(method="POST", data={"captured_image": "invaliddata"}):
        image, filename = load_image_from_capture()
        assert image is None
        assert filename is None


def test_load_image_from_file_no_input(monkeypatch):
    with app.test_request_context(method="POST", data={}):
        image, filename = load_image_from_file()
        assert image is None
        assert filename is None

def test_load_image_from_capture_no_input(monkeypatch):
    with app.test_request_context(method="POST", data={}):
        image, filename = load_image_from_capture()
        assert image is None
        assert filename is None


# pylint: disable=too-few-public-methods
class DummyInsertResult:
    """Dummy result to simulate MongoDB's insert_one return value."""

    inserted_id = "dummy_id"


def fake_insert_one(_doc):
    """Fake insert_one function that simulates storing a document."""
    return DummyInsertResult()


def fake_requests_post_success(_url, _json, _timeout):
    """Fake requests.post function that returns a successful fake response."""
    return FakeResponse({"results": {"dummy": "result"}}, status_code=200)

def fake_requests_post_error(_url, **kwargs):
    raise RequestException("Simulated ML client error")



def test_process_upload_oversize(monkeypatch):
    """
    Test that process_upload returns None if the processed image exceeds MAX_IMAGE_SIZE,
    and that it flashes an error.
    """
    img = Image.new("RGB", (5000, 5000), color="blue")
    flash_messages = []
    monkeypatch.setattr("src.app.flash", flash_messages.append)
    monkeypatch.setattr("src.app.MAX_IMAGE_SIZE", 1)
    result = process_upload(img, "big.jpg")
    assert result is None
    assert any("exceeds" in msg for msg in flash_messages)


def test_get_image_invalid_objectid(monkeypatch):
    """
    Test GET /uploads/ with an invalid ObjectId and verify that it redirects to the home page.
    """
    with app.test_client() as client:
        response = client.get("/uploads/invalid-id", follow_redirects=True)
        assert response.status_code == 200
        assert b"File Upload" in response.data

def test_load_image_from_capture_format_error(monkeypatch):
    with app.test_request_context(method="POST", data={"captured_image": "invaliddata"}):
        image, filename = load_image_from_capture()
        assert image is None
        assert filename is None

from requests import RequestException


def fake_requests_post_invalid_json(_url, **kwargs):
    class FakeResp:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            raise ValueError("Invalid JSON")
    return FakeResp()


def test_process_upload_ml_request_exception(monkeypatch):
    monkeypatch.setattr("src.app.requests.post", fake_requests_post_error)
    img = Image.new("RGB", (10, 10), "red")
    result = process_upload(img, "test.jpg")
    assert result is not None or "Error during prediction" in str(result)

def test_process_upload_json_error(monkeypatch):
    monkeypatch.setattr("src.app.requests.post", fake_requests_post_invalid_json)
    img = Image.new("RGB", (10, 10), "red")
    result = process_upload(img, "test.jpg")
    assert result is not None or "Error decoding ML response" in str(result)

def test_get_image_invalid_id(monkeypatch):
    with app.test_client() as client:
        response = client.get("/uploads/invalid-id", follow_redirects=True)
        assert response.status_code == 200
        assert b"File Upload" in response.data

def test_load_image_from_file_valid(monkeypatch):
    # Create a simple valid JPEG image
    img = Image.new("RGB", (10, 10), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    # Wrap the BytesIO object with FileStorage for proper simulation
    fs = FileStorage(stream=img_bytes, filename="valid.jpg", content_type="image/jpeg")
    with app.test_request_context(method="POST", data={"image": fs}):
        image, filename = load_image_from_file()
        assert image is not None, "Expected a valid image object, but got None."
        assert filename == "valid.jpg", "Filename did not match the expected value."

def test_load_image_from_capture_valid(monkeypatch):
    # Create a valid image and encode it to base64
    img = Image.new("RGB", (10, 10), color="green")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"
    with app.test_request_context(method="POST", data={"captured_image": data_uri}):
        image, filename = load_image_from_capture()
        assert image is not None, "Expected a valid image object from capture, but got None."
        assert filename == "captured.jpg", "Expected filename 'captured.jpg' for capture image."

def test_process_upload_rgba(monkeypatch):
    # Create an RGBA image (with transparency)
    img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
    captured_doc = {}
    # Patch the insert_one function of images_collection to capture the stored doc
    def fake_insert_one(doc):
        nonlocal captured_doc
        captured_doc = doc
        class DummyResult:
            inserted_id = "dummy_id"
        return DummyResult()
    monkeypatch.setattr("src.app.images_collection.insert_one", fake_insert_one)
    result = process_upload(img, "rgba.png")
    assert result == "dummy_id"
    # Verify that the stored document includes non-empty image data
    assert "data" in captured_doc
    assert len(captured_doc["data"]) > 0

def test_get_image_success(monkeypatch):
    valid_id = "507f1f77bcf86cd799439011"  # valid 24-character hex string
    fake_doc = {
       "_id": valid_id,
       "data": b"dummy image data",
       "content_type": "image/jpeg",
       "filename": "test.jpg",
    }
    class FakeColl:
        def find_one(self, query):
            return fake_doc
    monkeypatch.setattr("src.app.images_collection", FakeColl())
    with app.test_client() as client:
       response = client.get(f"/uploads/{valid_id}")
       assert response.status_code == 200
       assert response.mimetype == "image/jpeg"


def test_get_image_db_exception(monkeypatch):
    def fake_find_one(query):
         from bson.errors import InvalidId
         raise InvalidId("Simulated invalid id")
    monkeypatch.setattr("src.app.images_collection", type("FakeColl", (), {"find_one": fake_find_one}))
    with app.test_client() as client:
         response = client.get("/uploads/someid", follow_redirects=True)
         assert response.status_code == 200
         # Check that the redirected page contains a marker from your home or upload template.
         # This assumes your home (index.html) template contains "File Upload" or similar.
         assert b"File Upload" in response.data or b"Main Page" in response.data

def test_get_image_send_file_exception(monkeypatch):
    # Define a fake send_file that always raises an exception.
    def fake_send_file(*args, **kwargs):
        raise Exception("Simulated send_file error")
    monkeypatch.setattr("src.app.send_file", fake_send_file)
    
    # Prepare a fake document that get_image will try to fetch.
    fake_doc = {
        "_id": "507f1f77bcf86cd799439011",
        "data": b"dummy",
        "content_type": "image/jpeg",
        "filename": "test.jpg",
    }
    
    # Use a fake collection with a bound find_one method.
    class FakeColl:
        def find_one(self, query):
            return fake_doc
    monkeypatch.setattr("src.app.images_collection", FakeColl())
    
    with app.test_client() as client:
         # Request the image (this will trigger send_file exception)
         response = client.get("/uploads/507f1f77bcf86cd799439011", follow_redirects=True)
         # Check that the response status is 200 (after redirection)
         assert response.status_code == 200
         # Since the exception is caught, get_image should redirect to upload.
         # Check that the resulting page contains an indicator of the upload page.
         # For example, our upload.html starts with an <h1>Upload Image</h1>
         assert b"Upload Image" in response.data or b"File Upload" in response.data
