import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
from math import sqrt

from one_hot_dna import one_hot_dna_torch


class DatasetTM(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, i):
        return (self.X_data[i], self.Y_data[i])


class SequentialTM(nn.Module):
    def __init__(self, X_size, dropout=0.2):
        self.X_size = X_size
        self.dropout = dropout

        super(SequentialTM, self).__init__()
        self.linear1 = nn.Linear(X_size, 64)
        self.linear2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(self.dropout)
        return

    def forward(self, x):
        # x = x.view(self.X_size, 1)
        x = self.dropout(F.relu(self.linear1(x)))
        y_pred = self.linear2(x)
        return y_pred


def train(model, dataloader, criterion, optimizer, n_epochs=10):
    train_losses = []
    for epoch in range(n_epochs):
        running_loss = 0
        # for i in tqdm(
        #     range(len(dataloader)), desc=f"Epoch {epoch+1}/{n_epochs}", ncols=60
        # ):
        for data in dataloader:
            x, y = data

            inputs = Variable(x)
            labels = Variable(y)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_losses.append(float(loss.item()))
        print("[Epoch %d] loss: %.3f" % (epoch + 1, running_loss / len(dataloader)))
    # print(f"Epoch {epoch+1}, loss = {loss.item()}")
    return train_losses


def predict(model, x):
    outputs = model(Variable(x))
    predicted = outputs.data
    return predicted


def test(model, X, Y):
    diff_list = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        pred = predict(model, x)
        diff_list.append(abs(float(pred - y)))
    print("RMSD = ", sqrt(sum([i ** 2 for i in diff_list]) / len(diff_list)))
    return


if __name__ == "__main__":
    df = pd.read_csv("training_data.csv", header=None)
    dataset = df.values
    X_seq = dataset[:, 0]
    max_seq_len = max([len(seq) for seq in X_seq])

    X = torch.FloatTensor([one_hot_dna_torch(seq, max_seq_len) for seq in X_seq])
    Y = torch.FloatTensor(list(dataset[:, 1]))

    X_train = X[:4000]
    Y_train = Y[:4000]
    X_test = X[4000:]
    X_test_seq = X_seq[4000:]
    Y_test = Y[4000:]

    TrainDataset = DatasetTM(X_train, Y_train)
    TrainDataloader = DataLoader(TrainDataset, batch_size=125, shuffle=True)

    TM_model = SequentialTM(X_size=max_seq_len * 4, dropout=0.01)
    TM_model.train()

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(TM_model.parameters(), lr=1)
    train(TM_model, TrainDataloader, criterion, optimizer, n_epochs=100)
    test(TM_model, X_test, Y_test)
