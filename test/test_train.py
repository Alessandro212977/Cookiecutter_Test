# flake8: noqa
import sys

sys.path.insert(1, "./src/models")

import os

import pytest
from train_model import train


@pytest.mark.skipif(
    not os.path.exists("data/processed/train_dataset.pt"), reason="data files not found"
)
def test_training():
    train()
    assert os.path.exists("./models/checkpoint.pth")
