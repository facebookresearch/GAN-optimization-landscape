# Copyright (c) ElementAI and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Script to train DCGAN on MNIST, adaptted from https://github.com/pytorch/examples/blob/master/dcgan/main.py"""

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn.utils import spectral_norm
from tensorboardX import SummaryWriter
import json
from gan_eval_metrics import mnist_inception_score
from lib.optim import ExtraAdam
import numpy as np
from torch.utils.data import Subset
from plot_path_tools import compute_path_stats, plot_path_stats, compute_eigenvalues,\
    plot_eigenvalues
import time
import pickle
from lib import models
import torch.nn.functional as F


def load_mnist(batchSize, imageSize=32, train=True, workers=2, dataroot='./data', subset=None):

    dataset = dset.MNIST(root=dataroot, train=train, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(imageSize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
                         ]))
    if subset is not None:
        idx = np.arange(len(dataset))
        np.random.RandomState(123).shuffle(idx)
        dataset = Subset(dataset, idx[:subset])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=True, num_workers=int(workers))
    return dataloader


def normalize_module2D(module, norm, dim):
    """
    Applies normalization `norm` to `module`.
    Optionally uses `dim`
    Returns a list of modules.
    """

    if norm == 'none':
        return [module]
    elif norm == 'batch':
        return [module, nn.BatchNorm2d(dim)]
    elif norm == 'instance':
        return [module, nn.InstanceNorm2d(dim)]
    elif norm == 'layer':
        return [module, nn.GroupNorm(1, dim)]
    elif norm == 'spectral':
        return [spectral_norm(module)]
    else:
        raise NotImplementedError('normalization [%s] is not found' % norm)


class Generator(nn.Module):
    def __init__(self, ngpu, nc, ngf, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(models.Discriminator):
    def __init__(self, ngpu, nc, ndf, norm='spectral', sigmoid=True):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.norm = norm
        self.sigmoid = sigmoid
        # NOTE: made a special cose for BN because we don't normalize first layer
        # I kept it this way to be able to load pre-trained models
        if self.norm != 'batch':
            self.main = nn.Sequential(
                *normalize_module2D(nn.Conv2d(nc, ndf, 4, 2, 1, bias=True), norm, ndf),
                nn.LeakyReLU(0.2, inplace=True),

                *normalize_module2D(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True), norm, ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                *normalize_module2D(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True), norm, ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=True),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=True),
            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        if self.sigmoid:
            output = torch.sigmoid(output)
        return output.view(-1, 1).squeeze(1)


def weights_init(m):
    """ custom weights initialization called on netG and netD """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_model_loss(config):
    """returns dis/gen loss functions based on the model"""
    if config.model == 'dcgan':
        return dcgan_loss_dis, dcgan_loss_gen
    elif config.model in ['wgan', 'wgan_gp']:
        return wgan_loss_dis, wgan_loss_gen
    # elif config.model == 'wgan_gp':
    #     return functools.partial(wgan_loss_dis, grad_penalty=True, gp_lambda=config.gp_lambda), wgan_loss_gen
    else:
        raise NotImplementedError('%s model is not implemented!' % config.model)


# def dcgan_loss_dis(x_real, x_fake, netD, device):
#     p_real, p_gen = netD(x_real), netD(x_fake)

#     criterion = nn.BCELoss()
#     real_label = torch.full((p_real.size(0),), 1, device=device)
#     fake_label = torch.full((p_real.size(0),), 0, device=device)
#     errD_real = criterion(p_real, real_label)
#     errD_gen = criterion(p_gen, fake_label)
#     dis_loss = errD_real + errD_gen
#     return dis_loss, p_real, p_gen


# def dcgan_loss_gen(x_fake, netD, device):
#     p_gen = netD(x_fake)

#     criterion = nn.BCELoss()
#     real_label = torch.full((p_gen.size(0),), 1, device=device)
#     gen_loss = criterion(p_gen, real_label)
#     return gen_loss, p_gen


def dcgan_loss_dis(x_real, x_fake, netD, device):
    p_real, p_gen = netD(x_real), netD(x_fake)
    dis_loss = F.softplus(-p_real).mean() + F.softplus(p_gen).mean()
    return dis_loss, p_real, p_gen


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
    # if grad_penalty:
    #     dis_loss += gp_lambda * netD.get_penalty(x_real.detach(), x_fake.detach())
    return dis_loss, score_real, score_gen



def main(config):
    print("Hyper-params:")
    print(config)

    # create exp folder and save config
    exp_dir = os.path.join(config.exp_dir, config.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    plots_dir = os.path.join(exp_dir, 'extra_plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    np.random.seed(config.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.manualSeed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {0!s}".format(device))

    dataloader = load_mnist(config.batchSize)
    eval_dataloader = load_mnist(config.batchSize, subset=5000)
    eig_dataloader = load_mnist(1000, train=True, subset=1000)
    fixed_noise = torch.randn(64, config.nz, 1, 1, device=device)

    # define the model
    netG = Generator(config.ngpu, config.nc, config.ngf, config.nz).to(device)
    netG.apply(weights_init)
    if config.netG != '':
        print('loading generator from %s' % config.netG)
        netG.load_state_dict(torch.load(config.netG)['state_gen'])
    print(netG)

    # sigmoid = config.model == 'dcgan'
    sigmoid = False

    netD = Discriminator(config.ngpu, config.nc, config.ndf, config.dnorm, sigmoid).to(device)
    netD.apply(weights_init)
    if config.netD != '':
        print('loading discriminator from %s' % config.netD)
        netD.load_state_dict(torch.load(config.netD)['state_dis'])
    print(netD)

    # evaluation G and D
    evalG = Generator(config.ngpu, config.nc, config.ngf, config.nz).to(device)
    evalG.apply(weights_init)
    evalD = Discriminator(config.ngpu, config.nc, config.ndf, config.dnorm, sigmoid).to(device)
    evalD.apply(weights_init)

    # defining the loss function
    model_loss_dis, model_loss_gen = define_model_loss(config)

    # # defining learning rates based on the model
    # if config.model in ['wgan', 'wgan_gp']:
    #     config.lrG = config.lrD / config.n_critic
    #     warnings.warn('modifying learning rates to lrD=%f, lrG=%f' % (config.lrD, config.lrG))

    if config.lrG is None:
        config.lrG = config.lrD

    # setup optimizer
    if config.optimizer == 'adam':
        optimizerD = optim.Adam(netD.parameters(), lr=config.lrD, betas=(config.beta1, config.beta2))
        optimizerG = optim.Adam(netG.parameters(), lr=config.lrG, betas=(config.beta1, config.beta2))
    elif config.optimizer == 'extraadam':
        optimizerD = ExtraAdam(netD.parameters(), lr=config.lrD)
        optimizerG = ExtraAdam(netG.parameters(), lr=config.lrG)

    elif config.optimizer == 'rmsprop':
        optimizerD = optim.RMSprop(netD.parameters(), lr=config.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=config.lrG)

    elif config.optimizer == 'sgd':
        optimizerD = optim.SGD(netD.parameters(), lr=config.lrD, momentum=config.beta1)
        optimizerG = optim.SGD(netG.parameters(), lr=config.lrG, momentum=config.beta1)
    else:
        raise ValueError('Optimizer %s not supported' % config.optimizer)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)

    summary_writer = SummaryWriter(log_dir=exp_dir)

    global_step = 0
    torch.save({'state_gen': netG.state_dict(),
                'state_dis': netD.state_dict()},
               '%s/checkpoint_step_%06d.pth' % (exp_dir, global_step))

    # compute and save eigen values function
    def comp_and_save_eigs(step, n_eigs=20):
        eig_checkpoint = torch.load('%s/checkpoint_step_%06d.pth' % (exp_dir, step),
                                    map_location=device)
        evalG.load_state_dict(eig_checkpoint['state_gen'])
        evalD.load_state_dict(eig_checkpoint['state_dis'])
        gen_eigs, dis_eigs, game_eigs = \
            compute_eigenvalues(evalG, evalD, eig_dataloader, config,
                                model_loss_gen, model_loss_dis,
                                device, verbose=True, n_eigs=n_eigs)
        np.savez(os.path.join(plots_dir, 'eigenvalues_%d' % step),
                 gen_eigs=gen_eigs, dis_eigs=dis_eigs, game_eigs=game_eigs)

        return gen_eigs, dis_eigs, game_eigs

    if config.compute_eig:
        # eigenvalues of initialization
        gen_eigs_init, dis_eigs_init, game_eigs_init = comp_and_save_eigs(0)

    for epoch in range(config.niter):
        for i, data in enumerate(dataloader, 0):
            global_step += 1
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            x_real = data[0].to(device)
            batch_size = x_real.size(0)
            noise = torch.randn(batch_size, config.nz, 1, 1, device=device)
            x_fake = netG(noise)

            errD, D_x, D_G_z1 = model_loss_dis(x_real, x_fake.detach(), netD, device)

            # gradient penalty
            if config.model == 'wgan_gp':
                errD += config.gp_lambda * netD.get_penalty(x_real.detach(), x_fake.detach())

            errD.backward()
            D_x = D_x.mean().item()
            D_G_z1 = D_G_z1.mean().item()

            if config.optimizer == "extraadam":
                if i % 2 == 0:
                    optimizerD.extrapolation()
                else:
                    optimizerD.step()
            else:
                optimizerD.step()

            # weight clipping
            if config.model == 'wgan':
                for p in netD.parameters():
                    p.data.clamp_(-config.clip, config.clip)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            if config.model == 'dcgan' or (config.model in ['wgan', 'wgan_gp'] and i % config.n_critic == 0):
                netG.zero_grad()
                errG, D_G_z2 = model_loss_gen(x_fake, netD, device)
                errG.backward()
                D_G_z2 = D_G_z2.mean().item()

                if config.optimizer == "extraadam":
                    if i % 2 == 0:
                        optimizerG.extrapolation()
                    else:
                        optimizerG.step()
                else:
                    optimizerG.step()

            if global_step % config.printFreq == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, config.niter, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                summary_writer.add_scalar("loss/D", errD.item(), global_step)
                summary_writer.add_scalar("loss/G", errG.item(), global_step)
                summary_writer.add_scalar("output/D_real", D_x, global_step)
                summary_writer.add_scalar("output/D_fake", D_G_z1, global_step)

        # every epoch save samples
        fake = netG(fixed_noise)
        # vutils.save_image(fake.detach(),
        #                   '%s/fake_samples_step-%06d.png' % (exp_dir, global_step),
        #                   normalize=True)
        fake_grid = vutils.make_grid(fake.detach(), normalize=True)
        summary_writer.add_image("G_samples", fake_grid, global_step)

        # generate samples for IS evaluation
        IS_fake = []
        for i in range(10):
            noise = torch.randn(500, config.nz, 1, 1, device=device)
            IS_fake.append(netG(noise))
        IS_fake = torch.cat(IS_fake)

        IS_mean, IS_std = mnist_inception_score(IS_fake, device)
        print("IS score: mean=%.4f, std=%.4f" % (IS_mean, IS_std))
        summary_writer.add_scalar("IS_mean", IS_mean, global_step)

        # do checkpointing
        checkpoint = {'state_gen': netG.state_dict(),
                      'state_dis': netD.state_dict()}
        torch.save(checkpoint, '%s/checkpoint_step_%06d.pth' % (exp_dir, global_step))
        last_chkpt = '%s/checkpoint_step_%06d.pth' % (exp_dir, global_step)

        if epoch == 0:
            # last_chkpt = '%s/checkpoint_step_%06d.pth' % (exp_dir, 0)  # for now
            checkpoint_1 = torch.load(last_chkpt, map_location=device)

            if config.compute_eig:
                # compute eigenvalues for epoch 1, just in case
                gen_eigs_curr, dis_eigs_curr, game_eigs_curr = comp_and_save_eigs(global_step)

        # if (epoch + 1) % 10 == 0:
        if global_step > 30000 and epoch % 5 == 0:
            checkpoint_2 = torch.load(last_chkpt, map_location=device)
            print("Computing path statistics...")
            t = time.time()

            hist = compute_path_stats(evalG, evalD, checkpoint_1, checkpoint_2, eval_dataloader,
                                      config, model_loss_gen, model_loss_dis, device, verbose=True)

            with open("%s/hist_%d.pkl" % (plots_dir, global_step), 'wb') as f:
                pickle.dump(hist, f)

            plot_path_stats(hist, plots_dir, summary_writer, global_step)

            print("Took %.2f minutes" % ((time.time() - t) / 60.))

        if config.compute_eig and global_step > 30000 and epoch % 10 == 0:
            # compute eigenvalues and save them
            gen_eigs_curr, dis_eigs_curr, game_eigs_curr = comp_and_save_eigs(global_step)

            plot_eigenvalues([gen_eigs_init, gen_eigs_curr], [dis_eigs_init, dis_eigs_curr],
                             [game_eigs_init, game_eigs_curr],
                             ['init', 'step_%d' % global_step], plots_dir, summary_writer,
                             step=global_step)


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataroot', default='./data', help='path to dataset')
        parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
        parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
        parser.add_argument('--printFreq', type=int, default=50, help='# updates before each print')

        parser.add_argument('--model', type=str, default='dcgan', choices=['dcgan', 'wgan', 'wgan_gp'],
                            help='model type of GAN model')
        parser.add_argument('--n_critic', type=int, default=5, help='number of critic updates per generator update (wgan/wgan_gp)')
        parser.add_argument('--gp_lambda', type=int, default=10, help='weight for gradient penalty (wgan_gp)')
        parser.add_argument('--clip', type=float, default=0.01, help='weight clip range (wgan)')

        parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--ndf', type=int, default=64)
        parser.add_argument('--nc', type=int, default=1)
        parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
        parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0002')
        parser.add_argument('--lrG', type=float, default=None, help='learning rate, default=0.0002 -- same as lrD')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.999')
        parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'extraadam', 'sgd', 'rmsprop'],
                            help='training optimizer')

        parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        parser.add_argument('--netG', default='', help="path to netG (to continue training)")
        parser.add_argument('--netD', default='', help="path to netD (to continue training)")
        parser.add_argument('--dnorm', default='spectral', choices=['batch', 'spectral', 'none', 'instance', 'layer'], help="Discriminator normalization")
        parser.add_argument('--exp_dir', type=str, default='EXP', help='directory of experiment')
        parser.add_argument('--exp_name', type=str, default='debug', help='directory of experiment')
        parser.add_argument('--manualSeed', type=int, help='manual seed')
        parser.add_argument('--compute_eig', type=int, choices=[0, 1], default=0)
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    main(Config().parse_args())
