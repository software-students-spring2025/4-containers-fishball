"""
Test module for src.app endpoints and functionality.
"""

import io
import os
import pytest
from PIL import Image
from bson.objectid import ObjectId
from requests import RequestException
from src.app import (
    load_image_from_request,
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
        response = client.post("/", data={}, follow_redirects=True)
        assert response.status_code == 200
        assert b"Image Upload & Capture" in response.data


def test_index_post_empty_filename():
    """Test POST with an empty file name returns a 'No file selected' error."""
    with app.test_client() as client:
        data = {"image": (io.BytesIO(b"dummy data"), "")}
        response = client.post("/", data=data, follow_redirects=True)
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
        response = client.post("/", data=data, follow_redirects=True)
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
            "/",
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
        assert b"Image Upload & Capture" in response.data


def test_load_image_from_request_file_invalid(monkeypatch):
    """Test that load_image_from_request returns (None, None)
    when an invalid image file is provided."""
    fake_file = io.BytesIO(b"not an image")
    fake_file.filename = "invalid.jpg"

    with app.test_request_context(method="POST", data={"image": fake_file}):
        image, filename = load_image_from_request()
        assert image is None
        assert filename is None


def test_load_image_from_request_captured_invalid(monkeypatch):
    """Test that load_image_from_request returns (None, None)
    when captured image data is invalid."""
    with app.test_request_context(
        method="POST", data={"captured_image": "invaliddata"}
    ):
        image, filename = load_image_from_request()
        assert image is None
        assert filename is None


def test_load_image_from_request_no_input(monkeypatch):
    """Test that load_image_from_request returns (None, None)
    when neither file nor captured image is provided."""
    with app.test_request_context(method="POST", data={}):
        image, filename = load_image_from_request()
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


def fake_requests_post_error(_url, _json, _timeout):
    """Fake requests.post function that simulates an error response by raising RequestException."""
    raise RequestException("ML client error")


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
    Test GET /uploads/ with an invalid ObjectId
    and verify that it flashes an error and redirects.
    """
    with app.test_client() as client:
        response = client.get("/uploads/invalid-id", follow_redirects=True)
        assert response.status_code == 200
        assert (
            b"Error retrieving image" in response.data
            or b"Image not found" in response.data
        )
