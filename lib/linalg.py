# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

from __future__ import print_function
import scipy.sparse.linalg as linalg
import torch
from torch import autograd
import numpy as np

class JacobianVectorProduct(linalg.LinearOperator):
    def __init__(self, grad, params):
        if isinstance(grad, (list, tuple)):
            grad = list(grad)
            for i, g in enumerate(grad):
                grad[i] = g.view(-1)
            self.grad = torch.cat(grad)
        elif isinstance(grad, torch.Tensor):
            self.grad = grad.view(-1)
        self.shape = (self.grad.size(0), self.grad.size(0))
        self.dtype = np.dtype('Float32')
        self.params = params

    def _matvec(self, v):
        v = torch.Tensor(v)
        if self.grad.is_cuda:
            v = v.cuda()
        grad_vector_product = torch.dot(self.grad, v)
        hv = autograd.grad(grad_vector_product, self.params, retain_graph=True, allow_unused=True)
        _hv = []
        for g, p in zip(hv, self.params):
            if g is None:
                g = torch.zeros_like(p)
            _hv.append(g.contiguous().view(-1))
        hv = torch.cat(_hv)
        return hv.cpu()

def test_hessian_eigenvalues():
    SIZE = 4
    params = torch.rand(SIZE, requires_grad=True)
    loss = (params**2).sum()/2
    grad = autograd.grad(loss, params, create_graph=True)[0]
    A = JacobianVectorProduct(grad, params)
    e = linalg.eigsh(A, k=2)
    return e

def test_jacobian_eigenvalues():
    SIZE = 4
    param_1 = torch.rand(SIZE, requires_grad=True)
    param_2 = torch.rand(SIZE, requires_grad=True)
    loss_1 = (param_1*param_2).sum()
    loss_2 = -(param_1*param_2).sum()
    grad_1 = autograd.grad(loss_1, param_1, create_graph=True)[0]
    grad_2 = autograd.grad(loss_2, param_2, create_graph=True)[0]
    grad = torch.cat([grad_1, grad_2])
    params =[param_1, param_2]
    A = JacobianVectorProduct(grad, params)
    e = linalg.eigs(A, k=2)
    return e

if __name__ == '__main__':
    print(test_hessian_eigenvalues())
    print(test_jacobian_eigenvalues())
