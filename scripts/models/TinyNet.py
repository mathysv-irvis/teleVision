import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..utils.constant import IM_SIZE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TinyNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.name = "tinynet"

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        dummy = torch.zeros(1, 3, IM_SIZE, IM_SIZE)
        x = self.pool(F.relu(self.conv1(dummy)))
        x = self.pool(F.relu(self.conv2(x)))
        flatten_size = x.numel()
        self.fc1 = nn.Linear(flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return  self.fc3(x)

