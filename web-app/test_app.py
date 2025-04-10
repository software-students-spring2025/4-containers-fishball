import io
import os

os.environ.setdefault("SECRET_KEY", "test_secret_key")
os.environ.setdefault("MONGO_DBNAME", "test_db")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/test_db")
os.environ.setdefault("ML_CLIENT_URL", "http://fake-ml-client")

import pytest
from PIL import Image
from bson.objectid import ObjectId
from src.app import app

def generate_valid_objectid():
    return str(ObjectId())


###############################################################################
# Fake Collection for Overriding MongoDB Operations in Tests
###############################################################################


class FakeImagesCollection:
    def __init__(self):
        self.data = {}

    def insert_one(self, doc):
        doc["_id"] = "test_id"
        self.data["test_id"] = doc

        class DummyResult:
            inserted_id = "test_id"

        return DummyResult()

    def find(self):
        return list(self.data.values())

    def find_one(self, query):
        _id = query.get("_id")
        return self.data.get(_id, None)


###############################################################################
# Pytest Fixtures
###############################################################################


@pytest.fixture(autouse=True)
def fake_images_collection(monkeypatch):
    fake_collection = FakeImagesCollection()
    monkeypatch.setattr("src.app.images_collection", fake_collection)
    return fake_collection


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("Bad response")

    def json(self):
        return self._json


@pytest.fixture
def fake_requests_post(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse({"result": "happy"})

    monkeypatch.setattr("src.app.requests.post", fake_post)


###############################################################################
# Test Cases
###############################################################################


def test_index_get():
    """Test that the GET '/' endpoint returns a 200 OK and renders a page."""
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert b"Upload" in response.data or b"upload" in response.data


def test_index_post_no_file():
    """Test POST without a file returns an error flash and redirect."""
    with app.test_client() as client:
        response = client.post("/", data={}, follow_redirects=True)
        assert response.status_code == 200
        assert b"Image Upload & Capture" in response.data


def test_index_post_empty_filename():
    """Test POST with an empty file name returns an error."""
    with app.test_client() as client:
        data = {"image": (io.BytesIO(b"dummy data"), "")}
        response = client.post("/", data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"No file selected" in response.data


def test_index_post_invalid_image(monkeypatch):
    """Test POST with an invalid image (simulating a PIL error)."""
    fake_file = io.BytesIO(b"not an image")
    fake_file.name = "invalid.jpg"
    with app.test_client() as client:
        data = {"image": (fake_file, "invalid.jpg")}
        response = client.post("/", data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Invalid image format!" in response.data


def test_index_post_success(fake_requests_post):
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
    """Test that a GET to a non-existent image ID redirects to index."""
    non_existent_id = "000000000000000000000000"
    with app.test_client() as client:
        response = client.get(f"/uploads/{non_existent_id}", follow_redirects=True)
        assert response.status_code == 200
        assert b"Image Upload & Capture" in response.data
