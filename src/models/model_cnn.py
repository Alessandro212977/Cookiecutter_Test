import torch
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current device: {}".format(device))

class MyAwesomeModel(nn.Module):
    def __init__(self, drop_p=0.5):
        """ Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        """
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, 
                                             out_channels=16, 
                                             kernel_size=5, 
                                             stride=1, 
                                             padding=2), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.out = nn.Linear(32*7*7, 10)

        #self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """ Forward pass through the network, returns the output logits """
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        
        return F.log_softmax(x, dim=1)


def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        #images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(
    model, trainloader, testloader, criterion, optimizer=None, epochs=5, print_every=20
):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    steps = 0
    running_loss = 0
    plot_data = []
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            #images.resize_(images.size()[0], 784)

            optimizer.zero_grad()
            #print("forwards size {}".format(images.size()))
            output = model.forward(images)
            #print("OUTPUT", output.size(), labels.size())
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
                )

                plot_data.append(running_loss / print_every)

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()
    plt.plot(plot_data)
    plt.show()
    plt.savefig("./reports/figures/loss_curve_ep_{}".format(epochs))
