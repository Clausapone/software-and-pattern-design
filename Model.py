import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sigmoid


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x)).squeeze(1)
        return x
