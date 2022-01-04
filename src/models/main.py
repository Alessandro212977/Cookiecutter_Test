import argparse
import sys

import model as mdl
import torch
from model import MyAwesomeModel
from torch import nn

from data import mnist


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel(784, 10, [256])

        train_set, test_set = mnist()
        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=256, shuffle=True
        )
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
        torch.save(checkpoint, "checkpoint.pth")

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here

        model = load_checkpoint(args.load_model_from)
        _, test_set = mnist()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = MyAwesomeModel(
        checkpoint["input_size"], checkpoint["output_size"], checkpoint["hidden_layers"]
    )

    model.load_state_dict(checkpoint["state_dict"])
    return model


if __name__ == "__main__":
    TrainOREvaluate()
