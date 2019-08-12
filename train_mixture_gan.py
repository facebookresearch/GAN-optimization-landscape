# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
import time
from tensorboardX import SummaryWriter
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from torch.utils.data import TensorDataset, DataLoader

from lib import  models
from lib import utils
from plot_path_tools import compute_path_stats, plot_path_stats, compute_eigenvalues, plot_eigenvalues

parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('-s', '--seed', default=1234, type=int)
parser.add_argument('-m', '--method', default="extra", choices=("sim", "alt", "extra"))
parser.add_argument('-lrg', '--learning-rate-gen', default=1e-2, type=float)
parser.add_argument('-lrd', '--learning-rate-dis', default=1e-1, type=float)
parser.add_argument('-c', '--clip', default=1, type=float)
parser.add_argument('--ema', default=0, type=float)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--model', type=str, default='gan', choices=['nsgan', 'gan', 'wgan', 'wgan_gp'],
                            help='model type of GAN model')
parser.add_argument('-gp', '--gp-lambda', type=float, default=1e-3, help='weight for gradient penalty (wgan_gp)')
parser.add_argument('--saving-frequency', type=int, default=1000)
parser.add_argument('--save-stats', action="store_true")
args = parser.parse_args()

CUDA = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHOD = args.method
DIM_HIDDEN_GEN = 100
DIM_HIDDEN_DIS = 100
DIM_DATA = 1
BATCH_SIZE = 100
CLIP = args.clip
LR_GEN = args.learning_rate_gen
LR_DIS = args.learning_rate_dis
SAVING_FREQUENCY = args.saving_frequency
NUM_ITER = 100000
SEED = args.seed
SIGMA = 1.
DETERMINISTIC = args.deterministic
BETA_EMA = args.ema
NUM_SAMPLES = 10000
MODEL = args.model
GRADIENT_PENALTY = args.gp_lambda
DIM_LATENT = args.nz
torch.manual_seed(SEED)
np.random.seed(1234)
OUTPUT_PATH = os.path.join(args.output, '%s_%.0e/%s_lrg=%.0e_lrd=%.0e/s=%i/%i'%(
                                MODEL, GRADIENT_PENALTY, METHOD, LR_GEN, LR_DIS, SEED, int(time.time())))

writer = SummaryWriter(log_dir=os.path.join(OUTPUT_PATH, 'run'))

if not os.path.exists(os.path.join(OUTPUT_PATH, 'checkpoints')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'checkpoints'))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.network = nn.Sequential(nn.Linear(DIM_LATENT, DIM_HIDDEN_GEN),
                                    nn.ReLU(),
                                    nn.Linear(DIM_HIDDEN_GEN, DIM_DATA))

    def forward(self, x):
        return self.network(x.view(-1, DIM_LATENT))

class Discriminator(models.Discriminator):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(nn.Linear(DIM_DATA, DIM_HIDDEN_DIS),
                                    nn.ReLU(),
                                    nn.Linear(DIM_HIDDEN_DIS, 1))

    def forward(self, x):
        return self.network(x.view(-1, DIM_DATA))

gen = Generator()
dis = Discriminator()
gen_avg = Generator()
dis_avg = Discriminator()
gen_avg.load_state_dict(gen.state_dict())
dis_avg.load_state_dict(dis.state_dict())
if CUDA:
    gen = gen.cuda()
    dis = dis.cuda()
    gen_avg = gen_avg.cuda()
    dis_avg = dis_avg.cuda()

model_loss_dis, model_loss_gen = utils.define_model_loss(args)

class GaussianMixture:
    def __init__(self, p, mus, sigmas, seed=1234):
        self.state = np.random.RandomState(seed)
        self.p = p
        self.mus = mus
        self.sigmas = sigmas

    def sample(self, n):
        idx = self.state.choice(len(self.p), p=self.p, size=n)
        mu = self.mus[idx]
        sigma = self.sigmas[idx]
        x = self.state.normal(size=mu.shape)
        x = mu + sigma*x
        x = torch.tensor(x).float()
        return x

prob = (0.5, 0.5)
mus = np.repeat(np.array([2.,-2.]).reshape(2,1), DIM_DATA, axis=1)
sigmas = np.repeat(np.array([SIGMA,SIGMA]).reshape(2,1), DIM_DATA, axis=1)

gm = GaussianMixture(prob, mus, sigmas)
x_examples = gm.sample(NUM_SAMPLES)
z_examples = torch.zeros(NUM_SAMPLES, DIM_LATENT).normal_()
dataset = TensorDataset(torch.tensor(x_examples), torch.tensor(z_examples))
if DETERMINISTIC:
    np.savez(os.path.join(OUTPUT_PATH, 'data.npz'), x=x_examples, z=z_examples)
    dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES)
else:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

i = 0
n_dis_update = 0
n_gen_update = 0
gen_eigs1, dis_eigs1, game_eigs1 = compute_eigenvalues(gen, dis, dataloader, args, model_loss_gen, model_loss_dis, device, verbose=True)
for epoch in range(NUM_ITER):
    for x, z in dataloader:
        update_gen = False
        if CUDA:
            z = z.cuda()
            x = x.cuda()

        x_gen = gen(z)
        loss_dis, D_x, D_G_z1 = model_loss_dis(x, x_gen, dis, device)
        loss_gen, D_G_z2 = model_loss_gen(x_gen, dis, device)
        if MODEL == 'wgan_gp':
            penalty = dis.get_penalty(x, x_gen, mode="data").mean()
            loss_dis += GRADIENT_PENALTY*penalty

        if i%2 == 0 or METHOD == "sim":
            current_gen = []
            for p in gen.parameters():
                current_gen.append(p.data.clone())

            current_dis = []
            for p in dis.parameters():
                current_dis.append(p.data.clone())

            grad_dis = autograd.grad(loss_dis, dis.parameters(), retain_graph=True)
            n_clip_params = 0
            n_params = 0
            dis_grad_norm = 0

            for p, p_avg, g in zip(dis.parameters(), dis_avg.parameters(), grad_dis):
                p.data -= LR_DIS*g
                n_params += g.numel()
                if MODEL == "wgan":
                    g = -g

                    mask = ((abs(p)==CLIP).float()*p.sign()*g.sign() <= 0)
                    g = g*mask.float()
                    dis_grad_norm += (g**2).sum()
                    p.data.clamp_(-CLIP, CLIP)
                    n_clip_params += (abs(p.data)==CLIP).sum()
                else:
                    dis_grad_norm += (g**2).sum()
                if BETA_EMA:
                    beta = BETA_EMA
                else:
                    beta = (n_dis_update+1)/float(n_dis_update+2)
                p_avg.data = p.data.clone()*(1-beta) + p_avg.data.clone()*beta
            n_dis_update +=1


            if METHOD != "alt":
                grad_gen = autograd.grad(loss_gen, gen.parameters())
                gen_grad_norm = 0
                for p, p_avg, g in zip(gen.parameters(), gen_avg.parameters(), grad_gen):
                    p.data -= LR_GEN*g
                    if BETA_EMA:
                        beta = BETA_EMA
                    else:
                        beta = (n_gen_update+1)/float((n_gen_update+2))
                    p_avg.data = p.data*(1-beta) + p_avg.data*beta
                    gen_grad_norm += (g**2).sum()
                n_gen_update += 1
                update_gen = True

        if METHOD == "alt" and i%2 != 0:
            grad_gen = autograd.grad(loss_gen, gen.parameters())
            gen_grad_norm = 0
            for p, p_avg, g in zip(gen.parameters(), gen_avg.parameters(), grad_gen):
                p.data -= LR_GEN*g
                if BETA_EMA:
                    beta = BETA_EMA
                else:
                    beta = (n_gen_update+1)/float((n_gen_update+2))
                p_avg.data = p.data*(1-beta) + p_avg.data*beta
                gen_grad_norm += (g**2).sum()
            n_gen_update += 1
            update_gen = True

        if METHOD == "extra" and i%2 != 0:
            grad_dis = autograd.grad(loss_dis, dis.parameters(), retain_graph=True)
            for j, p in enumerate(dis.parameters()):
                p.data = current_dis[j] - LR_DIS*grad_dis[j]
                if MODEL is "wgan":
                    p.data.clamp_(-CLIP, CLIP)

            grad_gen = autograd.grad(loss_gen, gen.parameters())
            for j, p in enumerate(gen.parameters()):
                p.data = current_gen[j] - LR_GEN*grad_gen[j]

        if (n_gen_update-1)%SAVING_FREQUENCY == 0 and update_gen:
            torch.save({'state_gen': gen.state_dict(), 'state_dis': dis.state_dict(),
             'state_gen_avg': gen_avg.state_dict(), 'state_dis_avg': dis_avg.state_dict(),},
              os.path.join(OUTPUT_PATH, 'checkpoints/%i.state'%(n_gen_update)))

            if CUDA:
                x = x_examples.cuda()
                z = z_examples.cuda()

            x_gen_avg = gen_avg(z)
            loss_dis_avg, D_x, D_G_z1 = model_loss_dis(x, x_gen_avg, dis_avg, device)
            loss_gen_avg, D_G_z2 = model_loss_gen(x_gen_avg, dis_avg, device)
            if MODEL == "wgan_gp":
                penalty = dis_avg.get_penalty(x, x_gen_avg, mode="data").mean()
                loss_dis_avg += GRADIENT_PENALTY*penalty

            grad_dis = autograd.grad(loss_dis_avg, dis_avg.parameters(), retain_graph=True)
            n_clip_params_avg = 0
            dis_grad_norm_avg = 0
            for p, g in zip(dis_avg.parameters(), grad_dis):
                if MODEL is "wgan":
                    g = -g
                    mask = ((abs(p)==CLIP)*p.sign()*g.sign() <= 0)
                    g = g*mask
                    dis_grad_norm_avg += (g**2).sum()
                    p.data.clamp_(-CLIP, CLIP)
                    n_clip_params_avg += (abs(p.data)==CLIP).sum()
                else:
                    dis_grad_norm_avg += (g**2).sum()

            grad_gen = autograd.grad(loss_gen_avg, gen_avg.parameters())
            gen_grad_norm_avg = 0
            for g in grad_gen:
                gen_grad_norm_avg += (g**2).sum()


            log_likelihood = torch.zeros(1) #0.5 + torch.log(torch.exp(-0.5*torch.sum(((x_gen-mus[0].view(1,DIM_DATA))/sigmas[0].view(1,DIM_DATA))**2, -1))/(sigmas[0].prod()*np.sqrt((2*np.pi)**DIM_DATA)) + torch.exp(-0.5*torch.sum(((x_gen-mus[1].view(1,DIM_DATA))/sigmas[1].view(1,DIM_DATA))**2, -1))/(sigmas[1].prod()*np.sqrt((2*np.pi)**DIM_DATA)))
            print("Iter: %i/%i, Loss dis: %.2e, Loss gen: %.2e, Log-likelihood: %.2e, n_clip_params: %i/%i, Gen grad norm: %.2e, Dis grad norm: %.2e"%(
            n_gen_update, NUM_ITER, loss_dis, loss_gen, log_likelihood.data[0], n_clip_params, n_params, gen_grad_norm, dis_grad_norm))
            print("Averaging, Loss dis: %.2e, Loss gen: %.2e, Log-likelihood: %.2e, n_clip_params: %i/%i, Gen grad norm: %.2e, Dis grad norm: %.2e"%(
            loss_dis_avg, loss_gen_avg, 0, n_clip_params_avg, n_params, gen_grad_norm_avg, dis_grad_norm_avg))

            writer.add_scalar('log-likelihood', log_likelihood.data[0], n_gen_update)
            writer.add_scalar('loss-dis', loss_dis, n_gen_update)
            writer.add_scalar('loss-gen', loss_gen, n_gen_update)
            writer.add_scalar('n_clip_params', float(n_clip_params)/n_params, n_gen_update)
            writer.add_scalar('gen-grad-norm', gen_grad_norm, n_gen_update)
            writer.add_scalar('dis-grad-norm', dis_grad_norm, n_gen_update)

            writer.add_scalar('avg_loss-dis', loss_dis_avg, n_gen_update)
            writer.add_scalar('avg_loss-gen', loss_gen_avg, n_gen_update)
            writer.add_scalar('avg_n_clip_params', float(n_clip_params_avg)/n_params, n_gen_update)
            writer.add_scalar('avg_grad-norm-gen', gen_grad_norm_avg, n_gen_update)
            writer.add_scalar('avg_grad-norm-dis', dis_grad_norm_avg, n_gen_update)


            x_gen = gen(z)
            fig = plt.figure()
            plt.hist(x_gen.cpu().squeeze().data, bins=100)
            writer.add_figure('hist', fig, n_gen_update)
            plt.clf()

            fig = plt.figure()
            plt.hist(x_gen_avg.cpu().squeeze().data, bins=100)
            writer.add_figure('hist_avg', fig, n_gen_update)
            plt.clf()

            if args.save_stats:
                if n_gen_update == 1:
                    checkpoint_1 = torch.load(os.path.join(OUTPUT_PATH, 'checkpoints/%i.state'%(n_gen_update)), map_location=device)

                if n_gen_update > 1:
                    checkpoint_2 = torch.load(os.path.join(OUTPUT_PATH, 'checkpoints/%i.state'%(n_gen_update)), map_location=device)
                    hist = compute_path_stats(gen, dis, checkpoint_1, checkpoint_2, dataloader,
                                              args, model_loss_gen, model_loss_dis, device, verbose=True)

                    gen_eigs2, dis_eigs2, game_eigs2 = compute_eigenvalues(gen, dis, dataloader, args, model_loss_gen,
                    model_loss_dis, device, verbose=True, n_eigs=100)
                    hist.update({'gen_eigs':[gen_eigs1, gen_eigs2], 'dis_eigs':[dis_eigs1, dis_eigs2],
                                'game_eigs':[game_eigs1, game_eigs2]})

                    if not os.path.exists(os.path.join(OUTPUT_PATH, "extra_plots/data")):
                        os.makedirs(os.path.join(OUTPUT_PATH, "extra_plots/data"))
                    if not os.path.exists(os.path.join(OUTPUT_PATH, "extra_plots/plots")):
                        os.makedirs(os.path.join(OUTPUT_PATH, "extra_plots/plots"))

                    with open(os.path.join(OUTPUT_PATH, "extra_plots/data/%i.pkl"%(n_gen_update)), 'wb') as f:
                        pickle.dump(hist, f)

                    plot_path_stats(hist, os.path.join(OUTPUT_PATH,"extra_plots/plots"), writer, n_gen_update)
                    plot_eigenvalues([gen_eigs1, gen_eigs2], [dis_eigs1, dis_eigs2], [game_eigs1, game_eigs2],
                                ['init', 'step_%i'%i], os.path.join(OUTPUT_PATH,"extra_plots/plots"), writer, n_gen_update)

                    game_eigs2 = np.array(game_eigs2)
                    max_imag_eig = game_eigs2.imag.max()
                    max_real_eig = game_eigs2.real.max()
                    max_ratio = (game_eigs2.imag/game_eigs2.real).max()

                    writer.add_scalar('max_imag_eig', max_imag_eig, n_gen_update)
                    writer.add_scalar('max_real_eig', max_real_eig, n_gen_update)
                    writer.add_scalar('max_ratio', max_ratio, n_gen_update)
                    writer.add_scalar('max_gen_eig', gen_eigs2[-1], n_gen_update)
                    writer.add_scalar('max_dis_eig', dis_eigs2[-1], n_gen_update)


        i += 1
