# Copyright (c) ElementAI and its affiliates.
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Written by Amjad Almahairi (amjad.almahairi@elementai.com)

import torch
from torch import nn

from torch.nn import functional as F
import torch.utils.data
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np
from scipy.stats import entropy
import os


def load_mnist(batchSize, imageSize=32, train=True, workers=2, dataroot='./data'):

    dataset = dset.MNIST(root=dataroot, train=train, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(imageSize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
                         ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=True, num_workers=int(workers))
    return dataloader


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.25, training=self.training)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=0.25, training=self.training)
        out = self.fc3(out)
        return out


def train_mnist_classifier(lr=0.001, epochs=50, model_dir='.'):
    """train mnist classifier for inception score"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {0!s}".format(device))

    train_loader = load_mnist(batchSize=100, train=True)
    test_loader = load_mnist(batchSize=100, train=False)

    model = LeNet().to(device)

    def evaluate():
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy

    train_criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # training loop
    print('Started training...')
    best_test_acc = 0.0
    best_test_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).squeeze(1)
            loss = train_criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        test_acc = evaluate()
        print('Test Accuracy: {:.2f}\n'.format(test_acc))
        if test_acc > best_test_acc:
            best_test_epoch = epoch
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "mnist_classifier.pt"))

    print('Finished.')
    print('Best: Epoch: {}, Test-Accuracy: {:.4f}\n'.format(best_test_epoch, best_test_acc))


def mnist_inception_score(imgs, device=None, batch_size=500, splits=1, model_path='mnist_classifier.pt'):
    """Computes the inception score of the generated images imgs
    adapted from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

    imgs -- Torch tensor (bsx1x32x32) of images normalized in the range [-1, 1]
    device -- gpu/cpu
    batch_size -- batch size for feeding into mnist_classifier
    splits -- number of splits
    model_path -- path to pretrained mnist classifier
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Load inception model
    mnist_classifier = LeNet().to(device)
    mnist_classifier.load_state_dict(torch.load(model_path))
    mnist_classifier.eval()

    # Get predictions
    preds = np.zeros((N, 10))

    with torch.no_grad():
        i = 0
        total_i = 0
        while total_i < N:
            batch = imgs[i*batch_size:(i+1)*batch_size]
            batch_size_i = batch.size()[0]
            batch_preds = mnist_classifier(batch)
            batch_preds = F.softmax(batch_preds).cpu().numpy()
            preds[i*batch_size:i*batch_size + batch_size_i] = batch_preds
            i += 1
            total_i += batch_size_i

    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == "__main__":
    # train_mnist_classifier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {0!s}".format(device))

    train_loader = load_mnist(batchSize=5000, train=True)
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        is_mean, is_std = mnist_inception_score(imgs, device)
        print("mean-IS: %.4f, std-IS: %.2f" % (is_mean, is_std))
