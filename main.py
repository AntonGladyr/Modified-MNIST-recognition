#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn, optim
from time import time
from sklearn.model_selection import train_test_split
import pandas as pd
from cnn import Net
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

# standard deviation
STD = 0.5
MEAN = 0.5
INPUT_LAYER = 16384
H_LAYER_1 = 2048
H_LAYER_2 = 256
OUTPUT_LAYER = 10
# number of times we iterate over the training set
EPOCH = 15


def main():
    # load images as a numpy array
    train_dataset = np.array(np.load('/content/drive/My Drive/McGill/comp551/data/train_max_x', allow_pickle=True))
    train_dataset = train_dataset / 255.0
    train_dataset = train_dataset.astype('float32')
    targets = pd.read_csv('/content/drive/My Drive/McGill/comp551/data/train_max_y.csv', delimiter=',',
                          skipinitialspace=True)
    targets = targets.to_numpy()
    # remove id column
    targets = targets[:, 1]
    targets = targets.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(train_dataset, targets, test_size=0.2, random_state=42)
    # Clean memory
    train_dataset = None

    # converting training images into torch format
    dim1, dim2, dim3 = X_train.shape
    X_train = X_train.reshape(dim1, 1, dim2, dim3)
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)

    # converting validation images into torch format
    dim1, dim2, dim3 = X_test.shape
    X_test = X_test.reshape(dim1, 1, dim2, dim3)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # defining the model
    model = Net()

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    print(model)

    time0 = time()
    epochs = 1
    for e in range(epochs):
        model.train()
        running_loss = 0

        x_train, y_train = Variable(X_train).cuda(), Variable(y_train).cuda()
        x_val, y_val = Variable(X_test).cuda(), Variable(y_test).cuda()
        # converting the data into GPU format
        # if torch.cuda.is_available():
        #     x_train = x_train.cuda()
        #     y_train = y_train.cuda()
        #     x_val = x_val.cuda()
        #     y_val = y_val.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = model(x_train)
        output_val = model(x_val)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        loss_val = criterion(output_val, y_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss_train.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_dataset)))

    print("\nTraining Time (in minutes) =", (time() - time0) / 60)

    # prediction for validation set
    with torch.no_grad():
        output = model(x_val.cuda())

    ps = torch.exp(output).cpu()
    probab = list(ps.numpy())
    predictions = np.argmax(probab, axis=1)

    # accuracy on validation set

    print("\nModel Accuracy =", (accuracy_score(y_val, predictions)))


if __name__ == '__main__':
    main()