import pytest
import sys, os
import numpy as np

sys.path.insert(0, os.getcwd())

from src.utils import preproces_text, bag_of_words


def test_bag_of_words():
    all_words = ["hi", "how", "my", "name", "home", "shelter", "have", "live", "here", "tunisia"]
    assert list(bag_of_words(["i", "live", "in", "tunisia"], all_words)) == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
    ]


def test_preproces():
    text = "hello @,,, worldi I'm jhon ... i'm doing fine."
    assert preproces_text(text) == ["hello", "world", "i", "m", "jhon", "i", "m", "doing", "fin"]
