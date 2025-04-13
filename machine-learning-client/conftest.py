"""Conftest for the machine learning client tests.

This module injects a dummy deepface module for testing purposes.
"""

import sys
import types

import werkzeug

if not hasattr(werkzeug, "__version__"):
    werkzeug.__version__ = "2.3.3"

# Create a dummy module to stand in for deepface.
dummy_deepface = types.ModuleType("deepface")


# pylint: disable=too-few-public-methods
class DummyDeepFace:
    """A dummy DeepFace class for testing purposes.

    Provides a dummy implementation of analyze that returns a fixed analysis result.
    """

    @staticmethod
    def analyze(**kwargs):  # pylint: disable=unused-argument
        """Dummy analyze method that accepts any keyword arguments and returns a fixed result."""
        return {"emotion": {"happy": 1.0}}


dummy_deepface.DeepFace = DummyDeepFace

sys.modules["deepface"] = dummy_deepface

print("Injected dummy deepface module for testing purposes.")
