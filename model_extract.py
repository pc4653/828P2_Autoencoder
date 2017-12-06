from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
import random
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
import sys
import os

hidden_state_size = int(sys.argv[3])


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(96, 20)
        self.fc2 = nn.Linear(20, 20)		
        self.fc3 = nn.Linear(20, 20)		
        self.fc4 = nn.Linear(20, hidden_state_size)		
        self.fc7 = nn.Linear(hidden_state_size, 20)
        self.fc8 = nn.Linear(20, 20)
        self.fc9 = nn.Linear(20, 20)
        self.fc10 = nn.Linear(20, 96)
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return h

    def decode(self, z):
        h = self.relu(self.fc7(z))
        h = self.relu(self.fc8(h))
        h = self.relu(self.fc9(h))
        h = self.relu(self.fc10(h))
        return h

    def forward(self, x):
        #print(x)
        z = self.encode(x.view(-1, 96))
        return self.decode(z)
	
model = torch.load(sys.argv[1])
M = np.load(sys.argv[2])
if len(M) == 96:
    M = np.transpose(M)
temp = torch.from_numpy(M.astype(float))
temp = temp.float()
temp = Variable(temp)
temp = temp.cuda()
result = model(temp).cpu()
result = result.data.view(len(M), 96)
np.save(sys.argv[4], result.numpy())