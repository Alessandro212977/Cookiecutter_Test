import model_cnn as mdl
import torch
from torch import nn
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print("Training day and night")

# TODO: Implement training loop here
model = mdl.MyAwesomeModel()#MyAwesomeModel(784, 10, [256])

train_set, test_set = torch.load("data/processed/train_dataset.pt"), torch.load("data/processed/test_dataset.pt")
print("dataset loaded")

trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#print(type(images))
#print(images.shape)
#print(labels.shape)

testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
#dataiter = iter(testloader)
#images, labels = dataiter.next()
#print(type(images))
#print(images.shape)
#print(labels.shape)

try:
    state_dict = torch.load("models/checkpoint.pth")
    model.load_state_dict(state_dict)
except FileNotFoundError:
    print("No trained checkpoints found. Starting from scratch")

print("Training the model")
mdl.train(model, trainloader, testloader, nn.NLLLoss(), epochs=1)

print("Saving the model")
checkpoint = model.state_dict()
os.makedirs("models/checkpoint/", exist_ok=True)
torch.save(checkpoint, "models/checkpoint.pth")
