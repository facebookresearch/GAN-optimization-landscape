# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch.nn.functional as F

def compute_loss(p_true, p_fake, mode):
    if mode == 'nsgan':
        loss_gen = F.softplus(-p_fake).mean()
        loss_dis = F.softplus(-p_true).mean() + F.softplus(p_fake).mean()
    elif mode == 'gan':
        loss_gen = - F.softplus(-p_true).mean() - F.softplus(p_fake).mean()
        loss_dis = - loss_gen
    elif mode in ['wgan', 'wgan_gp']:
        loss_gen = p_true.mean() - p_fake.mean()
        loss_dis = - loss_gen
    else:
        raise NotImplementedError()
    return loss_gen, loss_dis

def define_model_loss(config):
    """returns dis/gen loss functions based on the model"""
    if config.model == 'gan':
        return dcgan_loss_dis, gan_loss_gen
    if config.model == 'nsgan':
        return dcgan_loss_dis, dcgan_loss_gen
    elif config.model in ['wgan', 'wgan_gp']:
        return wgan_loss_dis, wgan_loss_gen
    else:
        raise NotImplementedError('%s model is not implemented!' % config.model)

def dcgan_loss_dis(x_real, x_fake, netD, device):
    p_real, p_gen = netD(x_real), netD(x_fake)
    dis_loss = F.softplus(-p_real).mean() + F.softplus(p_gen).mean()
    return dis_loss, p_real, p_gen

def gan_loss_gen(x_fake, netD, device):
    p_gen = netD(x_fake)
    gen_loss = - F.softplus(p_gen).mean()
    return gen_loss, p_gen

def dcgan_loss_gen(x_fake, netD, device):
    p_gen = netD(x_fake)
    gen_loss = F.softplus(-p_gen).mean()
    return gen_loss, p_gen


def wgan_loss_gen(x_fake, netD, device):
    score_gen = netD(x_fake)
    gen_loss = -score_gen.mean()
    return gen_loss, score_gen


def wgan_loss_dis(x_real, x_fake, netD, device):
    score_real, score_gen = netD(x_real), netD(x_fake)
    dis_loss = score_gen.mean() - score_real.mean()
    return dis_loss, score_real, score_gen
