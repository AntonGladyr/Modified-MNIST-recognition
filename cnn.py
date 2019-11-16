import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # Defining a 2D convolution layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(6, 10, kernel_size=5, stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Defining another 2D convolution layer
        self.linear_layer1 = nn.Sequential(
            nn.Linear(5*5*16, 80)
        )
        self.linear_layer2 = nn.Sequential(
            nn.Linear(80, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        x = None
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear_layer1(out)
        out = self.linear_layer2(out)
        return out
