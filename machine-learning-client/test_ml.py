import io
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from app import app
from db_handler import DBHandler
from face_analyzer import FaceAnalyzer

# ----------------------
# Test face_analyzer.py
# ----------------------

def test_validate_config_valid():
    analyzer = FaceAnalyzer()
    assert analyzer.validate_config() is True

@patch("src.face_analyzer.DeepFace.analyze")
def test_analyze_success(mock_analyze):
    mock_analyze.return_value = {"emotion": {"happy": 0.9}}
    analyzer = FaceAnalyzer()
    result = analyzer.analyze("fake.jpg")
    assert result == {"emotion": {"happy": 0.9}}

@patch("src.face_analyzer.DeepFace.analyze", side_effect=ValueError("fail"))
def test_analyze_fail(mock_analyze):
    analyzer = FaceAnalyzer()
    result = analyzer.analyze("fake.jpg")
    assert result is None

# ----------------------
# Test db_handler.py
# ----------------------

@patch("src.db_handler.MongoClient")
def test_store_analysis(mock_client):
    mock_db = MagicMock()
    mock_client.return_value.__getitem__.return_value = mock_db
    mock_db.analyses.insert_one.return_value.inserted_id = "test_id"
    db = DBHandler()
    inserted_id = db.store_analysis("image.jpg", {"emotion": "happy"})
    assert inserted_id == "test_id"

@patch("src.db_handler.MongoClient")
def test_get_analysis(mock_client):
    mock_db = MagicMock()
    mock_client.return_value.__getitem__.return_value = mock_db
    mock_db.analyses.find_one.return_value = {"analysis_id": "abc123"}
    db = DBHandler()
    result = db.get_analysis("abc123")
    assert result == {"analysis_id": "abc123"}

# ----------------------
# Test app.py (Flask API)
# ----------------------

def test_analyze_no_file():
    with app.test_client() as client:
        response = client.post("/analyze", data={})
        assert response.status_code == 400
        assert "No file provided" in response.get_data(as_text=True)

@patch("src.app.analyzer.analyze", return_value=[{"emotion": {"happy": 0.95}}])
@patch("src.app.database.store_analysis", return_value="abc123")
def test_analyze_success(mock_store, mock_analyze):
    with app.test_client() as client:
        data = {
            "file": (io.BytesIO(b"fake image data"), "test.jpg")
        }
        response = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        assert "analysis_id" in response.get_json()

@patch("src.app.database.get_analysis", return_value={"analysis_id": "abc123", "results": {}})
def test_get_analysis_found(mock_get):
    with app.test_client() as client:
        response = client.get("/analysis/abc123")
        assert response.status_code == 200
        assert response.get_json()["analysis_id"] == "abc123"

@patch("src.app.database.get_analysis", return_value=None)
def test_get_analysis_not_found(mock_get):
    with app.test_client() as client:
        response = client.get("/analysis/unknown")
        assert response.status_code == 404
