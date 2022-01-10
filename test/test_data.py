import torch

train_set, test_set = torch.load("data/processed/train_dataset.pt"), torch.load("data/processed/test_dataset.pt")

assert len(train_set) == 20000 and len(test_set) == 5000

assert train_set[0][0].shape == torch.Size([1, 28, 28])

assert train_set[0][1].shape == torch.Size([])

labels = set()
for data in train_set:
    labels.add(data[1].item())
assert list(labels) == list(range(10))
