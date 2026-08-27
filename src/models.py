"""Model architectures exactly as designed; parameter counts asserted at construction."""
from __future__ import annotations

import torch.nn as nn

MNIST_PARAMS = 623_050
CIFAR_PARAMS = 1_674_570


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class MNISTCNN(nn.Module):
    """2 conv layers (1->32, 32->64), dropout 0.25, FC 3136->192, output 192->10. 623,050 params."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136, 192)
        self.fc2 = nn.Linear(192, 10)
        self.relu = nn.ReLU()
        assert count_params(self) == MNIST_PARAMS, count_params(self)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch_flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(self.drop(x))


class CIFAR10CNN(nn.Module):
    """3 conv layers (3->32, 32->64, 64->128), dropout 0.25, FC 2048->768, output 768->10. 1,674,570 params."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(2048, 768)
        self.fc2 = nn.Linear(768, 10)
        self.relu = nn.ReLU()
        assert count_params(self) == CIFAR_PARAMS, count_params(self)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch_flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(self.drop(x))


def torch_flatten(x):
    return x.view(x.size(0), -1)


def build_model(dataset: str) -> nn.Module:
    return MNISTCNN() if dataset == "mnist" else CIFAR10CNN()
