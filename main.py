from src.algorithms import bonferroni, holm_bonferroni, hochberg, benjamini_hochberg
from src.hypotheses import z_test
from src.utils import get_means, get_pvalues_z_test, get_t_statistics_z_test

import os

import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)

figure_1 = './figures/pvalues_h0.png'
figure_2 = './figures/pvalues_h1.png'

if not os.path.isfile(figure_1):
    m0 = 10000
    mu0 = 0
    mu0_ = np.ones(m0) * mu0
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    t_statistics = get_t_statistics_z_test(m0, 1, mu0_, mu0)
    pvalues = get_pvalues_z_test(t_statistics)
    ax.hist(pvalues, bins=10, range=(0, 1), density=True, edgecolor='black')
    ax.set_xlabel('p')
    ax.set_ylabel('density')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    fig.suptitle('Distribution of p-values under $H_0$')
    plt.tight_layout()
    plt.savefig(figure_1)

if not os.path.isfile(figure_2):
    m1 = 10000
    mu0 = 0
    mu1 = [1, 1.5, 2.5]
    n = [1, 5, 15]
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for i, n_ in enumerate(n):
        for j, mu1_ in enumerate(mu1):
            mu1s = np.ones(m1) * mu1_
            t_statistics = get_t_statistics_z_test(m1, n_, mu1s, mu0)
            pvalues = get_pvalues_z_test(t_statistics)
            ax[i][j].hist(pvalues, bins=10, range=(0, 1), density=True, edgecolor='black')
            ax[i][j].set_xlim(0, 1)
            ax[i][j].set_ylim(0, 1.1)
            if j == 0:
                ax[i][j].set_ylabel(f'$n = {n_}$')
            if i == 2:
                ax[i][j].set_xlabel(f'$\mu_1 = {mu1_}$')
    fig.suptitle('Distribution of p-values under $H_1$')
    plt.tight_layout()
    plt.savefig(figure_2)
