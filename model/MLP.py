'''
MLP
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, D_out=10):
        super().__init__()
        self.linear1 = nn.Linear(3072, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, D_out)

    def forward(self, x):
        x = x.view(-1,3*32*32)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.softmax(self.linear4(x))
        return x
