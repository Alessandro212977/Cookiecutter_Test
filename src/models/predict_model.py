import model_cnn as mdl
import torch

# load some images from the test dataset
print("Loading images")
test_set = torch.load("data/processed/test_dataset.pt")
testloader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=True)
dataiter = iter(testloader)
images, true_labels = dataiter.next()

# print(type(images))
# print(images.shape)
# print(true_labels.shape)

# load the model
print("Loading the model")
model = mdl.MyAwesomeModel()
try:
    state_dict = torch.load("models/checkpoint.pth")
    model.load_state_dict(state_dict)
except FileNotFoundError:
    print("Trained model not found, exiting...")
    quit()

# Predictions
ps = torch.exp(model.forward(images))
top_p, top_class = ps.topk(1, dim=1)
for i, label in enumerate(zip(true_labels, top_class.squeeze())):
    print("image {} is a {} and is predicted as a {}".format(i, *label))
