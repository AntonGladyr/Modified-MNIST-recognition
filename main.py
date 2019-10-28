#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn, optim
from time import time
from sklearn.model_selection import train_test_split
from PIL import Image

from torchvision import datasets, transforms


def main():
    train_dataset = np.array(np.load('data/train_max_x', allow_pickle=True))
    targets = np.genfromtxt('data/train_max_y.csv', delimiter=',', skip_header=1)
    targets = targets[:, 1]
    x = np.zeros((50000, 128, 128))
    print(x.shape)
    print(targets.shape)

    # TODO: fix
    x = np.expand_dims(x, axis=1)
    print(x.shape)
    np.insert(x, [1], targets.reshape(-1, 1), axis=1)
    # np.stack((x, targets), 1)
    print(x[0])
    print(x.shape)

    X_train, X_test, y_train, y_test = train_test_split(train_dataset, targets, test_size=0.2, random_state=42)

    print(X_train.shape)

    print(X_train.shape)

    model = nn.Sequential(nn.Linear(16384, 2048),
                          nn.ReLU(),
                          nn.Linear(2048, 10),
                          nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    images, labels = next(iter(train_dataset))
    images = images.view(images.shape[0], -1)
    logps = model(images)  # log probabilities
    loss = criterion(logps, labels)  # calculate the NLL loss

    print('Before backward pass: \n', model[0].weight.grad)
    loss.backward()
    print('After backward pass: \n', model[0].weight.grad)

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_dataset:
            # Flatten MNIST images into a 16384 long vector
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
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 16384)
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
    # img = Image.fromarray(data[1])
    #     img.show()


if __name__ == '__main__':
    main()
