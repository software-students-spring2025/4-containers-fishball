"""
Module for testing the machine learning client components.
"""

import io
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# pylint: disable=unused-import, import-error, wrong-import-position
from config import Config
from db_handler import DBHandler
from face_analyzer import FaceAnalyzer
from app import app


class TestFaceAnalyzer:
    """Test suite for the FaceAnalyzer class."""

    def test_validate_config_valid(self):
        """Verify that valid configuration returns True."""
        analyzer = FaceAnalyzer()
        assert analyzer.validate_config() is True

    @patch("src.face_analyzer.DeepFace.analyze")
    def test_analyze_success(self, mock_analyze):
        """Test that analyze() returns expected result when DeepFace.analyze works."""
        mock_analyze.return_value = {"emotion": {"happy": 0.9}}
        analyzer = FaceAnalyzer()
        result = analyzer.analyze("fake.jpg")
        assert result == {"emotion": {"happy": 0.9}}

    @patch("src.face_analyzer.DeepFace.analyze", side_effect=ValueError("fail"))
    def test_analyze_fail(self, _mock_analyze):
        """Test that analyze() returns None when an exception occurs."""
        analyzer = FaceAnalyzer()
        result = analyzer.analyze("fake.jpg")
        assert result is None


class TestDbHandler:
    """Test suite for the DBHandler class."""

    @patch("src.db_handler.uuid.uuid4")
    @patch("src.db_handler.MongoClient")
    def test_store_analysis(self, mock_client, mock_uuid):
        """Test that store_analysis returns the generated analysis_id."""

        class DummyUUID:
            """
            Dummy UUID class for testing purposes.
            Provides a dummy bytes property and a string representation.
            """

            @property
            def bytes(self):
                """Provide 16 dummy bytes."""
                return b"\x00" * 16

            def __str__(self):
                return "test_id"

        mock_uuid.return_value = DummyUUID()

        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_db.analyses.insert_one.return_value.inserted_id = "should_be_ignored"
        db_handler = DBHandler()
        inserted_id = db_handler.store_analysis("image.jpg", {"emotion": "happy"})
        assert str(inserted_id) == "test_id"

    @patch("src.db_handler.MongoClient")
    def test_get_analysis(self, mock_client):
        """Test that get_analysis returns the correct analysis document."""
        mock_db = MagicMock()
        analyses_mock = MagicMock()
        analyses_mock.find_one.return_value = {"analysis_id": "abc123"}
        mock_client.return_value.__getitem__.side_effect = lambda key: mock_db

        db_handler = DBHandler()
        db_handler.database.analyses = analyses_mock

        result = db_handler.get_analysis("abc123")
        assert result == {"analysis_id": "abc123"}


class TestApp:
    """Test suite for the Flask API endpoints in app.py."""

    @patch("src.app.images_collection")
    def test_analyze_no_file(self, _images_collection_mock):
        """Test that POST without file returns an error."""
        with app.test_client() as client:
            # Send an empty form (no file, no captured image)
            response = client.post("/", data={}, content_type="multipart/form-data")
            # Our app flashes error and redirects.
            # We assume a redirect status code (302) is returned.
            assert response.status_code == 302

    @patch("src.app.images_collection")
    def test_analyze_success(self, _images_collection_mock):
        """Test that POST with an image returns a redirect (i.e. analysis successful)."""
        with app.test_client() as client:
            # Create a dummy file as form data.
            data = {"image": (io.BytesIO(b"dummy data"), "test.jpg")}
            response = client.post("/", data=data, content_type="multipart/form-data")
            # Expect a redirect after POST (status code 302)
            assert response.status_code == 302

    @patch("src.app.images_collection")
    def test_get_image_not_found(self, images_collection_mock):
        """Test that GET /uploads/<id> when not found results in a redirect/error."""
        images_collection_mock.find_one.return_value = None
        with app.test_client() as client:
            response = client.get("/uploads/unknown")
            # Our app flashes an error and redirects in this case.
            # Accept a redirect status (302) or possibly a 404.
            assert response.status_code in (302, 404)


class TestAppJSON:
    """Additional tests for the Flask endpoints using JSON input."""

    @patch("src.app.analyzer")
    @patch("src.app.database")
    def test_json_no_image_field(self, _mock_database, _mock_analyzer):
        """Test POST with JSON but missing 'image' key returns a 400 error."""
        with app.test_client() as client:
            response = client.post("/", json={})
            # Expect a 400 error with an appropriate message.
            assert response.status_code == 400
            assert b"No file provided" in response.data

    @patch("src.app.analyzer")
    @patch("src.app.database")
    def test_json_unsupported_image_type(self, _mock_database, _mock_analyzer):
        """Test POST with JSON having an unsupported image type returns a 400 error."""
        data = {"image": "data:image/gif;base64,AAAA"}
        with app.test_client() as client:
            response = client.post("/", json=data)
            assert response.status_code == 400
            assert b"Unsupported image type" in response.data


# pylint: disable=too-few-public-methods
class TestAppAnalysisEndpoint:
    """Tests for the GET /analysis/<analysis_id> endpoint."""

    @patch("src.app.database")
    def test_get_analysis_not_found(self, mock_database):
        """Test that GET /analysis/<analysis_id> returns a 404 error when not found."""
        mock_database.get_analysis.return_value = None
        with app.test_client() as client:
            response = client.get("/analysis/unknown")
            assert response.status_code == 404
            json_resp = response.get_json()
            assert "error" in json_resp
