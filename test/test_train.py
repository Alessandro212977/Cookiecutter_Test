import sys
sys.path.insert(1, "./src/models")

from train_model import train
import os, pytest

@pytest.mark.skipif(not os.path.exists("data/processed/train_dataset.pt"), reason='data files not found')
def test_training():
    train()
    assert os.path.exists("./models/checkpoint.pth")