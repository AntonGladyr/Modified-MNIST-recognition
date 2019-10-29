#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn, optim
from time import time
from sklearn.model_selection import train_test_split

# standard deviation
STD = 0.5
MEAN = 0.5
INPUT_LAYER = 16384
H_LAYER_1 = 2048
H_LAYER_2 = 256
OUTPUT_LAYER = 10
# number of times we iterate over the training set
EPOCH = 15


def get_tensor_dataset(X, y):
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # scaled image values
    X = X / 255
    # image values range: [0, 1]
    X = ((X * STD) + MEAN)

    # to run on GPU
    X = X.cuda()
    y = y.cuda()

    return torch.utils.data.TensorDataset(X, y)  # create datset


def main():
    # load images as a numpy array
    train_dataset = np.array(np.load('data/train_max_x', allow_pickle=True))
    targets = np.genfromtxt('data/train_max_y.csv', delimiter=',', skip_header=1)
    # remove id column
    targets = targets[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(train_dataset, targets, test_size=0.2, random_state=42)

    # get_tensor_dataset by default returns data to run on GPU
    train_dataset = get_tensor_dataset(X_train, y_train)
    val_dataset = get_tensor_dataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset)

    model = nn.Sequential(nn.Linear(INPUT_LAYER, H_LAYER_1),
                          nn.ReLU(),
                          nn.Linear(H_LAYER_1, H_LAYER_2),
                          nn.ReLU(),
                          nn.Linear(H_LAYER_2, OUTPUT_LAYER),
                          nn.LogSoftmax(dim=1))
    model = model.cuda()  # to run on GPU

    criterion = nn.NLLLoss()
    images, labels = next(iter(train_loader))
    images = images.view(images.shape[0], -1)
    print(images)
    print('images shape: {0}'.format(images.shape))
    print('labels shape: {0}'.format(labels.shape))

    logps = model(images)  # log probabilities
    labels = labels.long()
    loss = criterion(logps, labels)  # calculate the NLL loss

    print('Before backward pass: \n', model[0].weight.grad)
    loss.backward()
    print('After backward pass: \n', model[0].weight.grad)

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = EPOCH
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Flatten images into a 16384 long vector
            labels = labels.long()
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_dataset)))
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)

    correct_count, all_count = 0, 0
    for images, labels in val_loader:
        for i in range(len(labels)):
            img = images[i].view(1, INPUT_LAYER)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))


if __name__ == '__main__':
    main()
