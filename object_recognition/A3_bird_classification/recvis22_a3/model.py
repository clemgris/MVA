import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152, resnet50, resnet101, ResNet152_Weights

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model_ft = resnet152(ResNet152_Weights.DEFAULT)
numFeatures = model_ft.fc.in_features
head = nn.Linear(numFeatures, nclasses)

#head = nn.Sequential(
#	nn.Linear(numFeatures, numFeatures),
#	nn.ReLU(),
#	nn.Dropout(0.25),
#	nn.Linear(numFeatures, numFeatures//2),
#	nn.ReLU(),
#	nn.Dropout(0.5),
#	nn.Linear(numFeatures//2, nclasses)
#)

model_ft.fc = head