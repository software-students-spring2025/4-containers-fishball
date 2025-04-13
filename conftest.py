# conftest.py
import sys
import types
import unittest.mock as mock

# Create a dummy module to stand in for deepface.
dummy_deepface = types.ModuleType("deepface")

# Create a dummy DeepFace class with a dummy analyze method.
class DummyDeepFace:
    @staticmethod
    def analyze(image_path):
        # Return a dummy result (adjust as needed for your tests)
        return {"emotion": {"happy": 1.0}}

dummy_deepface.DeepFace = DummyDeepFace

# Inject our dummy deepface into sys.modules so that any import of deepface gets this dummy.
sys.modules["deepface"] = dummy_deepface

print("Injected dummy deepface module for testing purposes.")
