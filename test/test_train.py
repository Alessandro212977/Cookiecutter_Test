import sys
sys.path.insert(1, "./src/models")

from train_model import train
import os

train()
assert os.path.exists("./models/checkpoint.pth")