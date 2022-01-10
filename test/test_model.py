import sys
sys.path.insert(1, "./src/models")

from model_cnn import MyAwesomeModel as mdl
import torch

model = mdl()

input = torch.zeros([1, 1, 28, 28])
output = model.forward(input)

assert output.shape == torch.Size([1, 10])