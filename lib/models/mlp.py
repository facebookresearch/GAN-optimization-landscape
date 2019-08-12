# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch.nn as nn
import torch.nn.functional as F
from .discriminator import Discriminator

class MLP_Generator(nn.Module):
    def __init__(self, n_in, n_out, layers=[256, 512, 1024], activation='lrelu', batchnorm=False, activation_out=None):
        super(MLP_Generator, self).__init__()

        network = []
        n_previous = n_in
        for l in layers:
            network.append(nn.Linear(n_previous, l))
            if batchnorm:
                network.append(nn.BatchNorm1d(l))
            if activation == 'relu':
                network.append(nn.ReLU())
            elif activation == 'lrelu':
                network.append(nn.LeakyReLU(0.2))
            n_previous = l
        network.append(nn.Linear(n_previous, n_out))

        self.network = nn.Sequential(*network)
        self.activation_out = activation_out

    def forward(self, z):
        x = self.network(z)
        if not self.activation_out is None:
            x = self.activation_out(x)
        return x

class MLP_Discriminator(Discriminator):
    def __init__(self, n_in, n_out, layers=[1024, 512, 256], activation='lrelu', batchnorm=False, dropout=0.):
        super(MLP_Discriminator, self).__init__()

        network = []
        n_previous = n_in
        for l in layers:
            network.append(nn.Linear(n_previous, l))
            if batchnorm:
                network.append(nn.BatchNorm1d(l))
            if activation == 'relu':
                network.append(nn.ReLU())
            elif activation == 'lrelu':
                network.append(nn.LeakyReLU(0.2))
            if dropout:
                network.append(nn.Dropout(dropout))
            n_previous = l
        network.append(nn.Linear(n_previous, n_out))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)
