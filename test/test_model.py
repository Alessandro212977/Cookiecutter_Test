import sys
sys.path.insert(1, "./src/models")
import pytest

from model_cnn import MyAwesomeModel as mdl
import torch

model = mdl()

input = torch.zeros([1, 1, 28, 28])
output = model.forward(input)

assert output.shape == torch.Size([1, 10]), "model doesn't produce the right output shape"

with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
    model(torch.randn(1, 2, 3))