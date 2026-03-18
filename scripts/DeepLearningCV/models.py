import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.name="net"

        self.backbone = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self,x):
        x = self.backbone(x)
        x = torch.flatten(x,1)
        return self.fc(x)


class TinyNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.name = "tinynet"
        self.conv = nn.Conv2d(3,16,3,padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self,x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x,1)
        return self.fc(x)

