# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
from .discriminator import Discriminator

class DCGAN64_Generator(nn.Module):
    def __init__(self, n_in, n_out, n_filters=128, activation='relu', batchnorm=True):
        super(DCGAN64_Generator, self).__init__()

        self.n_in = n_in

        self.deconv1 = nn.ConvTranspose2d(n_in, n_filters*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(n_filters*8)
        self.deconv2 = nn.ConvTranspose2d(n_filters*8, n_filters*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(n_filters*4)
        self.deconv3 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv4 = nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(n_filters)
        self.deconv5 = nn.ConvTranspose2d(n_filters, n_out, 4, 2, 1)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'lrelu':
            self.activation = lambda x: F.leaky_relu(x, 0.2)
        else:
            raise ValueError()

        self.batchnorm = batchnorm

    def forward(self, z):
        x = self.deconv1(z.view(-1,self.n_in,1,1))
        if self.batchnorm:
            x = self.deconv1_bn(x)
        x = self.activation(x)

        x = self.deconv2(x)
        if self.batchnorm:
            x = self.deconv2_bn(x)
        x = self.activation(x)

        x = self.deconv3(x)
        if self.batchnorm:
            x = self.deconv3_bn(x)
        x = self.activation(x)

        x = self.deconv4(x)
        if self.batchnorm:
            x = self.deconv4_bn(x)
        x = self.activation(x)

        x = F.tanh(self.deconv5(x))

        return x

class DCGAN64_Discriminator(Discriminator):
    def __init__(self, n_in, n_out, n_filters=128, activation='lrelu', batchnorm=True):
        super(DCGAN64_Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(n_in, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(n_filters*4)
        self.conv4 = nn.Conv2d(n_filters*4, n_filters*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(n_filters*8)
        self.conv5 = nn.Conv2d(n_filters*8, 1, 4, 1, 0)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'lrelu':
            self.activation = lambda x: F.leaky_relu(x, 0.2)
        else:
            raise ValueError()

        self.batchnorm = batchnorm

    def forward(self, x):
        x = self.activation(self.conv1(x))

        x = self.conv2(x)
        if self.batchnorm:
            x = self.conv2_bn(x)
        x = self.activation(x)

        x = self.conv3(x)
        if self.batchnorm:
            x = self.conv3_bn(x)
        x = self.activation(x)

        x = self.conv4(x)
        if self.batchnorm:
            x = self.conv4_bn(x)
        x = self.activation(x)

        x = self.conv5(x)

        return x

class DCGAN32_Generator(nn.Module):
    def __init__(self, n_in, n_out, n_filters=128, activation='relu', batchnorm=True):
        super(DCGAN32_Generator, self).__init__()

        self.n_in = n_in
        self.n_filters = n_filters

        self.deconv1 = nn.Linear(n_in, n_filters*4*4*4)
        self.deconv1_bn = nn.BatchNorm1d(n_filters*4*4*4)
        self.deconv2 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv3 = nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(n_filters)
        self.deconv5 = nn.ConvTranspose2d(n_filters, n_out, 4, 2, 1)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'lrelu':
            self.activation = lambda x: F.leaky_relu(x, 0.2)
        else:
            raise ValueError()

        self.batchnorm = batchnorm


    def forward(self, z):
        x = self.deconv1(z)
        if self.batchnorm:
            x = self.deconv1_bn(x)
        x = self.activation(x).view(-1,self.n_filters*4,4,4)

        x = self.deconv2(x)
        if self.batchnorm:
            x = self.deconv2_bn(x)
        x = self.activation(x)

        x = self.deconv3(x)
        if self.batchnorm:
            x = self.deconv3_bn(x)
        x = self.activation(x)

        x = F.tanh(self.deconv5(x))

        return x

class DCGAN32_Discriminator(Discriminator):
    def __init__(self, n_in, n_out, n_filters=128, activation='lrelu', batchnorm=True):
        super(DCGAN32_Discriminator, self).__init__()

        self.n_filters = n_filters

        self.conv1 = nn.Conv2d(n_in, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(n_filters*4)
        self.conv5 = nn.Linear(n_filters*4*4*4, n_out)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'lrelu':
            self.activation = lambda x: F.leaky_relu(x, 0.2)
        else:
            raise ValueError()

        self.batchnorm = batchnorm


    def forward(self, x):
        x = self.activation(self.conv1(x))
        #print x.size()

        x = self.conv2(x)
        #print x.size()
        if self.batchnorm:
            x = self.conv2_bn(x)
        x = self.activation(x)

        x = self.conv3(x)
        #print x.size()
        if self.batchnorm:
            x = self.conv3_bn(x)
        x = self.activation(x).view(-1, self.n_filters*4*4*4)

        x = self.conv5(x)

        return x

class DCGAN28_Generator(nn.Module):
    def __init__(self, n_in, n_out, n_filters=128, activation='relu', batchnorm=True):
        super(DCGAN28_Generator, self).__init__()

        self.n_in = n_in

        self.deconv1 = nn.ConvTranspose2d(n_in, n_filters*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(n_filters*8)
        self.deconv2 = nn.ConvTranspose2d(n_filters*8, n_filters*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(n_filters*4)
        self.deconv3 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 4, 2, 2)
        self.deconv3_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv4 = nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(n_filters)
        self.deconv5 = nn.ConvTranspose2d(n_filters, n_out, 3, 1, 1)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'lrelu':
            self.activation = lambda x: F.leaky_relu(x, 0.2)
        else:
            raise ValueError()

        self.batchnorm = batchnorm

    def forward(self, z):
        x = self.deconv1(z.view(-1,self.n_in,1,1))
        if self.batchnorm:
            x = self.deconv1_bn(x)
        x = self.activation(x)

        x = self.deconv2(x)
        if self.batchnorm:
            x = self.deconv2_bn(x)
        x = self.activation(x)

        x = self.deconv3(x)
        if self.batchnorm:
            x = self.deconv3_bn(x)
        x = self.activation(x)

        x = self.deconv4(x)
        if self.batchnorm:
            x = self.deconv4_bn(x)
        x = self.activation(x)

        x = F.tanh(self.deconv5(x))

        return x

class DCGAN28_Discriminator(Discriminator):
    def __init__(self, n_in, n_out, n_filters=128, activation='lrelu', batchnorm=True):
        super(DCGAN28_Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(n_in, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*4, 4, 2, 2)
        self.conv3_bn = nn.BatchNorm2d(n_filters*4)
        self.conv4 = nn.Conv2d(n_filters*4, n_filters*8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(n_filters*8)
        self.conv5 = nn.Conv2d(n_filters*8, 1, 3, 1, 0)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'lrelu':
            self.activation = lambda x: F.leaky_relu(x, 0.2)
        else:
            raise ValueError()

        self.batchnorm = batchnorm

    def forward(self, x):
        x = self.activation(self.conv1(x))

        x = self.conv2(x)
        if self.batchnorm:
            x = self.conv2_bn(x)
        x = self.activation(x)

        x = self.conv3(x)
        if self.batchnorm:
            x = self.conv3_bn(x)
        x = self.activation(x)

        x = self.conv4(x)
        if self.batchnorm:
            x = self.conv4_bn(x)
        x = self.activation(x)

        x = self.conv5(x)

        return x
