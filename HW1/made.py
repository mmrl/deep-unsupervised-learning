#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

rng = np.random.RandomState(1234214)

def sample2d(N, probs):
    values = np.asarray(list(product(range(200), range(200))))
    index = len(values)

    samples = rng.choice(index, size=N, replace=True, p=probs.reshape(-1))

    return values[samples]


probs = np.load('distribution.npy')
raw_data = sample2d(100000, probs=probs)

# plt.hist2d(raw_data[:, 0], raw_data[:, 1], bins=200)
# plt.savefig('../plots/2D-samples.pdf')
# plt.show()


def onehot_coding(data):
    N = len(data)
    a = np.zeros((N, 200, 2))

    for i in range(N):
        x1, x2 = data[i]
        a[i, x1, 0] = 1
        a[i, x2, 1] = 1

    return a

data = onehot_coding(raw_data)


def naive_softmax(logits):
    pred = np.exp(logits)
    return pred / pred.sum(axis=1)

def loss(pred, targets):
    loss = (pred * targets).sum(axis=1)
    loss = -np.log(loss)

    return loss.sum()

thetas = np.random.randn(200)

def prob_X1(data):
    # x1 =  data[:, :, 0]
    return naive_softmax(thetas)

weights = np.random.randn(5, 200)
def prob_X2(data):
    logits = data[:, :, 0]

    for w in weights[:-1]:
        logits = logits * w
        logits = np.sigmoid(logits)

    logits *= weights[-1]

    return naive_softmax(logits)

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(200))
        self.mlp = nn.Sequential(
            nn.Linear(200, 200),
            nn.Sigmoid(),
            nn.Linear(200, 200),
            nn.Sigmoid(),
            nn.Linear(200, 200),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, inputs):
        logp_x1 = F.log_softmax(
            self.theta.unsqueeze(0).expand((len(inputs), -1)), dim=1)
        logp_x2 = self.mlp(inputs[:, :, 0])

        return torch.stack([logp_x1, logp_x2], dim=2)

def create_dataset(raw_data, batch_size=100, test_split=0.2, val_split=0.2, onehot=True):
    N = len(raw_data)

    inputs = onehot_coding(raw_data) if onehot else raw_data
    targets = raw_data

    inputs = torch.from_numpy(inputs).to(dtype=torch.float32)
    targets = torch.from_numpy(targets).to(dtype=torch.long)

    dataset = TensorDataset(inputs, targets)

    test_size = int(N * test_split)
    val_size = int((N - test_size) * val_split)
    train_size = N - test_size - val_size

    training_data, validation_data, testing_data = random_split(
        dataset, [train_size, val_size, test_size])

    training_data = DataLoader(training_data, batch_size=batch_size)
    validation_data = DataLoader(validation_data, batch_size=batch_size)
    testing_data = DataLoader(testing_data, batch_size=batch_size)

    return training_data, validation_data, testing_data


def train(model, data, epochs=10):
    optimiser = optim.SGD(model.parameters(), 0.0001)
    loss_fn = nn.NLLLoss()

    training_data, validation_data = data

    trace = []

    for _ in range(epochs):
        for inputs, targets in training_data:
            pred = model(inputs)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimiser.step()

            trace.append(loss.item())
        print(trace[-1])

    return trace

model = BaselineModel()
data = create_dataset(raw_data)

trace = train(model, data[:2], 10)

plt.plot(trace)
plt.show()
