import model as mdl
import torch
from model import MyAwesomeModel
from torch import nn


print("Training day and night")

# TODO: Implement training loop here
model = MyAwesomeModel(784, 10, [256])

train_set, test_set = torch.load("../../data/processed/train_dataset.pt"), torch.load("../../data/processed/test_dataset.pt")
print("dataset loaded")
trainloader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)
dataiter = iter(testloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

mdl.train(model, train_set, test_set, nn.NLLLoss(), epochs=1)

checkpoint = {
    "input_size": 784,
    "output_size": 10,
    "hidden_layers": [each.out_features for each in model.hidden_layers],
    "state_dict": model.state_dict(),
}
torch.save(checkpoint, "../../models/checkpoint.pth")
