# Copyright (c) 2017 Ishaan Gulrajani
# Copyright (c) 2017 Marvin Cao
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

from torch.autograd import Variable, grad
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        raise NotImplementedError()

    def get_penalty(self, x_true, x_gen, mode="linear"):
        x_true = x_true.view_as(x_gen)
        if mode ==  "linear":
            alpha = torch.rand((len(x_true),)+(1,)*(x_true.dim()-1))
            if x_true.is_cuda:
                alpha = alpha.cuda(x_true.get_device())
            x_penalty = alpha*x_true + (1-alpha)*x_gen
        elif mode == "gen":
            x_penalty = x_gen.clone()
        elif mode == "data":
            x_penalty = x_true.clone()
        x_penalty.requires_grad_()
        p_penalty = self.forward(x_penalty)
        gradients = grad(p_penalty, x_penalty, grad_outputs=torch.ones_like(p_penalty).cuda(x_true.get_device()) if x_true.is_cuda else torch.ones_like(p_penalty), create_graph=True, retain_graph=True, only_inputs=True)[0]
        penalty = ((gradients.view(len(x_true), -1).norm(2, 1) - 1)**2).mean()

        return penalty
