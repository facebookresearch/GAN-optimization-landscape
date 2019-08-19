# Copyright (c) ElementAI and its affiliates.
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
import torch
import os
import matplotlib
import torchvision.utils as vutils
from gan_eval_metrics import mnist_inception_score
from torch import autograd
from matplotlib import gridspec
# from lib import utils
from lib.linalg import JacobianVectorProduct
import scipy.sparse.linalg as linalg

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_path_stats(gen, dis, checkpoint_1, checkpoint_2, dataloader, config,
                       model_loss_gen, model_loss_dis,
                       device=None, path_min=-0.1, path_max=1.1, n_points=100,
                       key_gen='state_gen', key_dis='state_dis', verbose=False):
    """
    Computes stats for plotting path between checkpoint_1 and checkpoint_2.

    Parameters
    ----------
    gen: Generator
    dis: Discriminator
    checkpoint_1: pytorch checkpoint
        first checkpoint to plot path interpolation
    checkpoint_2: pytorch checkpoint
        second checkpoint to plot path interpolation
    dataloader: pytorch DataLoader
        real data loader (mnist)
    config: Namespace
        configuration (hyper-parameters) for the generator/discriminator
    model_loss_dis, model_loss_gen: function
        returns generator and discriminator losses given the discriminator output
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We compute diff which is a vector representing the vector between input1 and input2
    # it is useful later when we compute the cosine similarity and dot product.
    params_diff = []
    for name, p in gen.named_parameters():
        d = (checkpoint_1[key_gen][name] - checkpoint_2[key_gen][name])
        params_diff.append(d.flatten())

    for name, p in dis.named_parameters():
        d = (checkpoint_1[key_dis][name] - checkpoint_2[key_dis][name])
        params_diff.append(d.flatten())

    params_diff = torch.cat(params_diff)

    # The different statistics we want to compute are saved in a dict.
    hist = {'alpha': [], 'cos_sim': [], 'dot_prod': [], 'gen_loss': [], 'dis_loss': [],
            'penalty': [], 'grad_gen_norm': [], 'grad_dis_norm': [], 'grad_total_norm': []}

    start_time = time.time()

    # Compute statistics we are interested in for different values of alpha.
    for alpha in np.linspace(path_min, path_max, n_points):

        ############### Computing and loading interpolation ##############
        # We compute the interpolation between input1 and input2
        # with interpolation-coefficient = alpha and load them into the model.
        # When alpha = 0 then the model is equal to the parameters of input1.
        state_dict_gen = gen.state_dict()
        for p in checkpoint_1[key_gen]:
            state_dict_gen[p] = alpha * checkpoint_2[key_gen][p] + (1 - alpha) * checkpoint_1[key_gen][p]
        gen.load_state_dict(state_dict_gen)

        state_dict_dis = dis.state_dict()
        for p in checkpoint_1[key_dis]:
            state_dict_dis[p] = alpha * checkpoint_2[key_dis][p] + (1 - alpha) * checkpoint_1[key_dis][p]
        dis.load_state_dict(state_dict_dis)

        gen = gen.to(device)
        dis = dis.to(device)
        #################################################################

        ######### Compute Loss and Gradient over Full-Batch ##########
        # cos_sim = 0
        # norm_grad_gen = 0
        # norm_grad_dis = 0
        # dot_prod = 0

        gen_loss_epoch = 0
        dis_loss_epoch = 0
        penalty_epoch = 0
        grad_gen_epoch = {}
        for name, param in gen.named_parameters():
            grad_gen_epoch[name] = torch.zeros_like(param).flatten()
        grad_dis_epoch = {}
        for name, param in dis.named_parameters():
            grad_dis_epoch[name] = torch.zeros_like(param).flatten()

        n_data = 0
        t0 = time.time()
        for i, x_true in enumerate(dataloader):
            x_true = x_true[0]
            z = torch.randn(x_true.size(0), config.nz, 1, 1)

            x_true = x_true.to(device)
            z = z.to(device)

            for p in gen.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            for p in dis.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            ################# Compute Loss #########################
            # TODO: Needs to be changed to be able to handle different kind of loss
            x_gen = gen(z)
            dis_loss, _, _ = model_loss_dis(x_true, x_gen.detach(), dis, device)
            gen_loss, _ = model_loss_gen(x_gen, dis, device)
            if config.model == 'wgan_gp':
                penalty = dis.get_penalty(x_true.detach(), x_gen.detach()).mean()
                dis_loss += config.gp_lambda * penalty
            else:
                penalty = torch.zeros(1)
            #################################################

            for p in dis.parameters():
                p.requires_grad = False
            gen_loss.backward(retain_graph=True)
            for p in dis.parameters():
                p.requires_grad = True

            for p in gen.parameters():
                p.requires_grad = False
            dis_loss.backward()
            for p in gen.parameters():
                p.requires_grad = True

            for name, param in gen.named_parameters():
                grad_gen_epoch[name] += param.grad.flatten() * len(x_true)

            for name, param in dis.named_parameters():
                grad_dis_epoch[name] += param.grad.flatten() * len(x_true)

            gen_loss_epoch += gen_loss.item() * len(x_true)
            dis_loss_epoch += dis_loss.item() * len(x_true)
            penalty_epoch += penalty.item() * len(x_true)
            n_data += len(x_true)
        ########################################################

        gen_loss_epoch /= n_data
        dis_loss_epoch /= n_data
        penalty_epoch /= n_data

        grad_gen = []
        for name, _ in gen.named_parameters():
            grad_gen.append(grad_gen_epoch[name])
        grad_dis = []
        for name, param in dis.named_parameters():
            param_flat = param.flatten()
            grad_param = grad_dis_epoch[name]
            if config.model == 'wgan':
                # zero-out gradient that violate wgan weight constraints
                zero_mask = (torch.abs(param_flat) == config.clip) &\
                            (torch.sign(grad_param) == torch.sign(param_flat))
                grad_param[zero_mask] = 0.0
            grad_dis.append(grad_param)

        grad_gen = torch.cat(grad_gen) / n_data
        grad_dis = torch.cat(grad_dis) / n_data
        grad_all = torch.cat([grad_gen, grad_dis])

        ####### Compute statistics we are interested in ##########
        # Compute squared norm of the gradient
        norm_grad_gen = (grad_gen**2).sum().cpu().numpy()
        norm_grad_dis = (grad_dis**2).sum().cpu().numpy()

        # Compute the dot product (unnormalized cosine similarity)
        dot_prod = (grad_all * params_diff).sum() / torch.sqrt((params_diff**2).sum())

        # Compute cosine similarity
        cos_sim = dot_prod / torch.sqrt((grad_all**2).sum())

        dot_prod = dot_prod.item()
        cos_sim = cos_sim.item()

        # # Compute cosine similarity
        # cos_sim = 1 - distance.cosine(grad_all, params_diff)

        # # Compute the dot product (unnormalized cosine similarity)
        # dot_prod = (grad_all * params_diff).sum() / np.sqrt((params_diff**2).sum())
        ##########################################################
        if verbose:
            print("Alpha: %.2f, Angle: %.2f, Generator loss: %.2e, Discriminator loss: %.2e, Penalty: %.2f, Gen grad norm: %.2e, Dis grad norm: %.2e, Time: %.2fsec"
                  % (alpha, cos_sim, gen_loss_epoch, dis_loss_epoch, penalty_epoch, norm_grad_gen, norm_grad_dis, time.time() - t0))

        hist['alpha'].append(alpha)
        hist['cos_sim'].append(cos_sim)
        hist['dot_prod'].append(dot_prod)
        hist['gen_loss'].append(gen_loss_epoch)
        hist['dis_loss'].append(dis_loss_epoch)
        hist['penalty'].append(penalty_epoch)
        hist['grad_gen_norm'].append(norm_grad_gen)
        hist['grad_dis_norm'].append(norm_grad_dis)
        hist['grad_total_norm'].append(norm_grad_dis + norm_grad_gen)

    if verbose:
        print("Time to finish: %.2f minutes" % ((time.time() - start_time) / 60.))

    return hist


def plot_path_stats(hist, out_dir=None, summary_writer=None, step=0):
    """
    plots interpolation path in `hist` and computed by `compute_path_stats`.
    """
    assert out_dir is not None or summary_writer is not None, 'save results either as files in out_dir or in tensorboard!'

    fig1 = plt.figure()
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1)
    # the fisrt subplot
    ax0 = plt.subplot(gs[0])
    cos_sim, = ax0.plot(hist['alpha'], hist['cos_sim'], color='r')

    # plt.plot(hist['alpha'], hist['cos_sim'])
    x_previous = hist['cos_sim'][0]
    for i, x in enumerate(hist['cos_sim']):
        if x * x_previous < 0:
            ax0.axvline(x=hist['alpha'][i], color='black', linestyle='--')
        x_previous = x
    ax0.axhline(y=0, color='black', linestyle='--')
    # the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_yscale('log')
    grad_norm, = ax1.plot(hist['alpha'], hist['grad_total_norm'])
    ax1.axvline(x=0, color='black', linestyle='--')
    ax1.axvline(x=1, color='black', linestyle='--')

    plt.setp(ax0.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    # put lened on first subplot
    ax0.legend((cos_sim, grad_norm), ('cos_sim', 'grad_total_norm'), loc='upper right')

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)

    fig2 = plt.figure()
    plt.plot(hist['alpha'], hist['dot_prod'])
    x_previous = hist['dot_prod'][0]
    for i, x in enumerate(hist['dot_prod']):
        if x * x_previous < 0:
            plt.axvline(x=hist['alpha'][i], color='black', linestyle='--')
        x_previous = x
    plt.axhline(y=0, color='black', linestyle='--')

    fig3 = plt.figure()
    plt.plot(hist['alpha'], hist['gen_loss'])
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axvline(x=1, color='black', linestyle='--')

    fig4 = plt.figure()
    plt.plot(hist['alpha'], hist['dis_loss'])
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axvline(x=1, color='black', linestyle='--')

    fig5 = plt.figure()
    plt.plot(hist['alpha'], hist['penalty'])
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axvline(x=1, color='black', linestyle='--')

    fig6 = plt.figure()
    plt.plot(hist['alpha'], hist['grad_gen_norm'])
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axvline(x=1, color='black', linestyle='--')
    plt.yscale('log')

    fig7 = plt.figure()
    plt.plot(hist['alpha'], hist['grad_dis_norm'])
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axvline(x=1, color='black', linestyle='--')
    plt.yscale('log')

    fig8 = plt.figure()
    grad_norm = np.sqrt(np.array(hist['grad_gen_norm']) + np.array(hist['grad_dis_norm']))
    y_coord = np.sqrt(abs(grad_norm**2 - np.array(hist['dot_prod'])**2))
    plt.quiver(hist['alpha'][::2], 0, hist['dot_prod'][::2], y_coord[::2], width=0.003, scale=np.max(grad_norm) * 2)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axvline(x=1, color='black', linestyle='--')
    plt.ylim(0, 1)
    plt.xlim(-0.5, 1.5)

    # fig9 = plt.figure()
    # plt.plot(hist['alpha'], hist['grad_total_norm'])
    # plt.axvline(x=0, color='black', linestyle='--')
    # plt.axvline(x=1, color='black', linestyle='--')
    # plt.yscale('log')

    if out_dir is not None:
        fig1.savefig(os.path.join(out_dir, 'cos_sim_%06d.png' % step))
        fig2.savefig(os.path.join(out_dir, 'dot_prod_%06d.png' % step))
        fig3.savefig(os.path.join(out_dir, 'gen_loss_%06d.png' % step))
        fig4.savefig(os.path.join(out_dir, 'dis_loss_%06d.png' % step))
        fig5.savefig(os.path.join(out_dir, 'penalty_%06d.png' % step))
        fig6.savefig(os.path.join(out_dir, 'grad_gen_norm_%06d.png' % step))
        fig7.savefig(os.path.join(out_dir, 'grad_dis_norm_%06d.png' % step))
        fig8.savefig(os.path.join(out_dir, 'grad_direction_%06d.png' % step))
        # fig9.savefig(os.path.join(out_dir, 'grad_total_norm_%06d.png' % step))

    if summary_writer is not None:
        summary_writer.add_figure('cos_sim', fig1, step)
        summary_writer.add_figure('dot_prod', fig2, step)
        summary_writer.add_figure('gen_loss', fig3, step)
        summary_writer.add_figure('dis_loss', fig4, step)
        summary_writer.add_figure('grad_gen_norm', fig6, step)
        summary_writer.add_figure('grad_dis_norm', fig7, step)
        # summary_writer.add_figure('grad_total_norm', fig9, step)
        summary_writer.add_figure('grad_direction', fig8, step)
        summary_writer.add_figure('penalty', fig5, step)


def visualize_interpolation(gen, checkpoint_1, checkpoint_2, config, out_dir, n_samples=64,
                            device=None, path_min=-0.1, path_max=1.1, n_points=100,
                            key_gen='state_gen', verbose=False):
    """
    Computes stats for plotting path between checkpoint_1 and checkpoint_2.

    Parameters
    ----------
    gen: Generator
    checkpoint_1: pytorch checkpoint
        first checkpoint to plot path interpolation
    checkpoint_2: pytorch checkpoint
        second checkpoint to plot path interpolation
    config: Namespace
        configuration (hyper-parameters) for the generator/discriminator
    out_dir: str
        path to output directory
    n_samples: int
        number of samples per interpolation point
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fixed_noise = torch.randn(n_samples, config.nz, 1, 1, device=device)
    samples = []
    IS_scores = []
    nrow = int(np.sqrt(n_samples))
    start_time = time.time()
    alpha_vals = []

    # Compute statistics we are interested in for different values of alpha.
    for alpha in np.linspace(path_min, path_max, n_points):
        alpha_vals.append(alpha)
        state_dict_gen = gen.state_dict()
        for p in checkpoint_1[key_gen]:
            state_dict_gen[p] = alpha * checkpoint_2[key_gen][p] + (1 - alpha) * checkpoint_1[key_gen][p]
        gen.load_state_dict(state_dict_gen)

        gen = gen.to(device)
        with torch.no_grad():
            fake = gen(fixed_noise)
            samples.append(fake.cpu())
            vutils.save_image(fake, '%s/samples-%f.png' % (out_dir, alpha), nrow=nrow, normalize=True)

            # generate samples for IS evaluation
            IS_fake = []
            for i in range(10):
                noise = torch.randn(500, config.nz, 1, 1, device=device)
                IS_fake.append(gen(noise))
            IS_fake = torch.cat(IS_fake)

            IS_mean, IS_std = mnist_inception_score(IS_fake, device)
            IS_scores.append(IS_mean)

            if verbose:
                print("IS score: alpha=%.2f, mean=%.4f, std=%.4f" % (alpha, IS_mean, IS_std))

    fig = plt.figure()
    plt.plot(alpha_vals, IS_scores)
    fig.savefig(os.path.join(out_dir, 'IS_scores_path.png'))

    results = {'samples': torch.cat(samples),
               'IS_scores': torch.tensor(IS_scores)}

    torch.save(results, '%s/vis_results.pt' % (out_dir))

    if verbose:
        print("Time to finish: %.2f minutes" % ((time.time() - start_time) / 60.))


def compute_eigenvalues(gen, dis, dataloader, config,
                        model_loss_gen, model_loss_dis,
                        device=None, n_eigs=20, verbose=False, imaginary=False):
    """
    Computes stats for plotting path between checkpoint_1 and checkpoint_2.

    Parameters
    ----------
    gen: Generator
    dis: Discriminator
    dataloader: pytorch DataLoader
        real data loader (mnist)
    config: Namespace
        configuration (hyper-parameters) for the generator/discriminator
    """

    start_time = time.time()

    grad_gen_epoch = [torch.zeros_like(p) for p in gen.parameters()]
    grad_dis_epoch = [torch.zeros_like(p) for p in dis.parameters()]
    n_data = 0
    for i, x_true in enumerate(dataloader):
        print(i)
        x_true = x_true[0]
        z = torch.randn(x_true.size(0), config.nz, 1, 1)

        x_true = x_true.to(device)
        z = z.to(device)

        ################# Compute Loss #########################
        # TODO: Needs to be changed to be able to handle different kind of loss
        x_gen = gen(z)
        dis_loss, _, _ = model_loss_dis(x_true, x_gen, dis, device)
        gen_loss, _ = model_loss_gen(x_gen, dis, device)
        # p_true, p_gen = dis(x_true), dis(x_gen)
        # gen_loss, dis_loss = utils.compute_loss(p_true, p_gen, mode=config.model)
        if config.model == 'wgan_gp':
            penalty = dis.get_penalty(x_true, x_gen).mean()
            dis_loss += config.gp_lambda * penalty
        else:
            penalty = torch.zeros(1)

        grad_gen = autograd.grad(gen_loss, gen.parameters(), create_graph=True)
        grad_dis = autograd.grad(dis_loss, dis.parameters(), create_graph=True)

        for i, g in enumerate(grad_gen):
            grad_gen_epoch[i] += g * len(x_true)

        for i, g in enumerate(grad_dis):
            grad_dis_epoch[i] += g * len(x_true)
        n_data += len(x_true)

    grad_gen_epoch = [g / n_data for g in grad_gen_epoch]
    grad_dis_epoch = [g / n_data for g in grad_dis_epoch]

    t0 = time.time()
    A = JacobianVectorProduct(grad_gen_epoch, list(gen.parameters()))
    if imaginary:
        gen_eigs = linalg.eigs(A, k=n_eigs, which='LI')[0]
    else:
        gen_eigs = linalg.eigsh(A, k=n_eigs)[0]
    print("Time to compute Eig-values: %.2f" % (time.time() - t0))

    t0 = time.time()
    A = JacobianVectorProduct(grad_dis_epoch, list(dis.parameters()))
    if imaginary:
        dis_eigs = linalg.eigs(A, k=n_eigs, which='LI')[0]
    else:
        dis_eigs = linalg.eigsh(A, k=n_eigs)[0]
    print("Time to compute Eig-values: %.2f" % (time.time() - t0))

    t0 = time.time()
    grad = grad_gen_epoch + grad_dis_epoch
    params = list(gen.parameters()) + list(dis.parameters())
    A = JacobianVectorProduct(grad, params)
    if imaginary:
        game_eigs = linalg.eigs(A, k=n_eigs, which='LI')[0]
    else:
        game_eigs = linalg.eigs(A, k=n_eigs)[0]
    print("Time to compute Eig-values: %.2f" % (time.time() - t0))

    if verbose:
        print(gen_eigs[:5])
        print(dis_eigs[:5])
        print(game_eigs[:5])
        print("Time to finish: %.2f minutes" % ((time.time() - start_time) / 60.))

    return gen_eigs, dis_eigs, game_eigs


def plot_eigenvalues(gen_eigs, dis_eigs, game_eigs, labels=None, out_dir=None, summary_writer=None, step=0):
    """
    plots interpolation path in `hist` and computed by `compute_path_stats`.
    """
    assert out_dir is not None or summary_writer is not None, 'save results either as files in out_dir or in tensorboard!'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fig1 = plt.figure()
    for i, eigs in enumerate(game_eigs):
        plt.scatter(eigs.real, eigs.imag, label=labels[i])
    plt.legend()

    fig2 = plt.figure()
    for i, eigs in enumerate(gen_eigs):
        plt.bar(np.arange(len(eigs)), eigs[::-1], label=labels[i])
    plt.legend()

    fig3 = plt.figure()
    for i, eigs in enumerate(dis_eigs):
        plt.bar(np.arange(len(eigs)), eigs[::-1], label=labels[i])
    plt.legend()

    if out_dir is not None:
        fig1.savefig(os.path.join(out_dir, 'game_eigs_%06d.png' % step))
        fig2.savefig(os.path.join(out_dir, 'gen_eigs_%06d.png' % step))
        fig3.savefig(os.path.join(out_dir, 'dis_eigs_%06d.png' % step))

    if summary_writer is not None:
        summary_writer.add_figure('game_eigs', fig1, step)
        summary_writer.add_figure('gen_eigs', fig2, step)
        summary_writer.add_figure('dis_eigs', fig3, step)
