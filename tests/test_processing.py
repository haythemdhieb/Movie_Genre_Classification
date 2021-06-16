import pytest
import sys, os

sys.path.append(os.path.dirname(__file__))
from src.utils import preproces_text


def test_preproces():
    text = "hello @,,, worldi I'm jhon ... i'm doing fine."
    assert preproces_text(text) == ["hello", "im"]
