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


input_file = sys.argv[1]
hidden_state_size = int(sys.argv[2])
trial_num = int(sys.argv[3])

print('hidden state size is ' + str(hidden_state_size))
print('trial #' + str(trial_num))
batch_size = 10
epochs = 20000
log_interval = 10
kwargs = {'num_workers': 1, 'pin_memory': True} 

train_loss_log = []
avg_train_loss_log = []
dset = np.load(input_file)
if len(dset) == 96:
    dset = np.transpose(dset)
#5 percent for testing, chosen 
test_size = int(0.05*len(dset))

train_data = []
test_data = []
for i in range(0,len(dset)-test_size):
    temp = torch.from_numpy(dset[i].astype(float))
    temp = temp.float()
    train_data.append(temp)

for i in range(len(dset)-test_size,len(dset)):
    temp = torch.from_numpy(dset[i].astype(float))
    temp = temp.float()
    test_data.append(temp)    

	
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)


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


model = AE()
model.cuda()


def loss_function(recon_x, x):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 96))
    MSE = F.mse_loss(recon_x, x.view(-1,96))
    return MSE


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        noise = np.random.poisson(np.random.uniform(1,5,1), data.numpy().shape)
        noise = torch.from_numpy(noise.astype(float))
        noise = noise.float()
        noisy_data = data + noise
        noisy_data = Variable(noisy_data)
        noisy_data = noisy_data.cuda()
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch= model(noisy_data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
            train_loss_log.append(train_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    avg_train_loss_log.append(train_loss / len(train_loader.dataset))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, data in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch= model(data)
        test_loss += loss_function(recon_batch, data).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

torch.save(model, input_file + '_' + str(hidden_state_size) + '_hidden_state_trial_' + str(trial_num) + '.pt')	
loss_index = sum(avg_train_loss_log[len(avg_train_loss_log)-100:len(avg_train_loss_log)])/100
		
import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.asarray(train_loss_log, dtype = np.float32))
plt.title('train_loss')
plt.savefig(input_file + '_' + str(hidden_state_size) + '_train_loss_trial_' + str(trial_num) + '.png')

plt.figure()
plt.plot(np.asarray(avg_train_loss_log, dtype = np.float32))
plt.title('avg_train_loss' + ' ' + str(loss_index))
plt.savefig(input_file + str(hidden_state_size) + '_avg_train_loss_trial_' + str(trial_num) + '.png')





