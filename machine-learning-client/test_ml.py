"""
Module for testing the machine learning client components.
"""

import io
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
# pylint: disable=wrong-import-position
from src.app import app
from src.db_handler import DBHandler
from src.face_analyzer import FaceAnalyzer

# ----------------------
# Test face_analyzer.py
# ----------------------
class TestFaceAnalyzer:
    """
    Test suite for the FaceAnalyzer class.
    """

    def test_validate_config_valid(self):
        """
        Test that validate_config returns True for a valid configuration.
        """
        analyzer = FaceAnalyzer()
        assert analyzer.validate_config() is True

    @patch("src.face_analyzer.DeepFace.analyze")
    def test_analyze_success(self, mock_analyze):
        """
        Test that the analyze method successfully returns expected results.
        """
        mock_analyze.return_value = {"emotion": {"happy": 0.9}}
        analyzer = FaceAnalyzer()
        result = analyzer.analyze("fake.jpg")
        assert result == {"emotion": {"happy": 0.9}}

    @patch("src.face_analyzer.DeepFace.analyze", side_effect=ValueError("fail"))
    def test_analyze_fail(self):
        """
        Test that the analyze method returns None when an exception occurs.
        """
        analyzer = FaceAnalyzer()
        result = analyzer.analyze("fake.jpg")
        assert result is None

# ----------------------
# Test db_handler.py
# ----------------------
class TestDbHandler:
    """
    Test suite for the DBHandler class.
    """

    @patch("src.db_handler.MongoClient")
    def test_store_analysis(self, mock_client):
        """
        Test that the store_analysis method stores data correctly in the database.
        """
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_db.analyses.insert_one.return_value.inserted_id = "test_id"
        db = DBHandler()
        inserted_id = db.store_analysis("image.jpg", {"emotion": "happy"})
        assert inserted_id == "test_id"

    @patch("src.db_handler.MongoClient")
    def test_get_analysis(self, mock_client):
        """
        Test that the get_analysis method retrieves the correct analysis from the database.
        """
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_db.analyses.find_one.return_value = {"analysis_id": "abc123"}
        db = DBHandler()
        result = db.get_analysis("abc123")
        assert result == {"analysis_id": "abc123"}

# ----------------------
# Test app.py (Flask API)
# ----------------------

class TestApp:
    """
    Test suite for the Flask API endpoints in app.py.
    """
    @patch("src.app.database.store_analysis", return_value=None)
    def test_analyze_no_file(self):
        """
        Test that the analyze endpoint returns a 400 error when no file is provided.
        """
        with app.test_client() as client:
            response = client.post("/analyze", data={})
            assert response.status_code == 400
            assert "No file provided" in response.get_data(as_text=True)

    def test_analyze_success(self):
        """
        Test that the analyze endpoint successfully processes and stores analysis data.
        """
        with app.test_client() as client:
            data = {
                "file": (io.BytesIO(b"fake image data"), "test.jpg")
            }
            response = client.post("/analyze", data=data, content_type="multipart/form-data")
            assert response.status_code == 200
            assert "analysis_id" in response.get_json()

    @patch("src.app.database.get_analysis", return_value={"analysis_id": "abc123", "results": {}})
    def test_get_analysis_found(self):
        """
        Test that the get_analysis endpoint returns the correct analysis when found.
        """
        with app.test_client() as client:
            response = client.get("/analysis/abc123")
            assert response.status_code == 200
            assert response.get_json()["analysis_id"] == "abc123"

    @patch("src.app.database.get_analysis", return_value=None)
    def test_get_analysis_not_found(self):
        """
        Test that the get_analysis endpoint returns a 404 error when the analysis is not found.
        """
        with app.test_client() as client:
            response = client.get("/analysis/unknown")
            assert response.status_code == 404
