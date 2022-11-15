from src.hypotheses import z_test
from src.utils import get_pvalues_z_test, get_t_statistics_z_test, FDP, FWE

import os

import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)

figure_1 = './figures/pvalues_h0.png'
figure_2 = './figures/pvalues_h1.png'
figures_3 = ['./figures/FWER_bonferroni.png', './figures/FWER_holm_bonferroni.png', './figures/FWER_hochberg.png', './figures/FWER_benjamini_hochberg.png']
figures_4 = ['./figures/FDR_bonferroni.png', './figures/FDR_holm_bonferroni.png', './figures/FDR_hochberg.png', './figures/FDR_benjamini_hochberg.png']

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
                ax[i][j].set_ylabel(fr'$n = {n_}$')
            if i == 2:
                ax[i][j].set_xlabel(fr'$\mu_1 = {mu1_}$')
    fig.suptitle('Distribution of p-values under $H_1$')
    plt.tight_layout()
    plt.savefig(figure_2)

if not os.path.isfile(figures_3[0]) or not os.path.isfile(figures_4[0]):
    alpha = 0.05
    mu0 = 0
    mu1 = [1, 1.5, 2.5]
    m0 = 500
    m1 = 500
    n = [1, 5, 15]
    reps = 10000

    figs3 = [plt.subplots(3, 3, figsize=(12, 12)) for i in range(4)]
    figs4 = [plt.subplots(3, 3, figsize=(12, 12)) for i in range(4)]
    for i, n_ in enumerate(n):
        for j, mu1_ in enumerate(mu1):
            results = z_test(mu0, mu1_, m0, m1, n=n_, reps=reps, alpha=alpha)
            for k, algorithm in enumerate(['bonferroni', 'holm_bonferroni', 'hochberg', 'benjamini_hochberg']):
                fwe = FWE(results['true'], results[algorithm])
                fwer = fwe.mean()
                ax = figs3[k][1][i][j]
                ax.hist(fwe, bins=2, range=(0, 1), density=True, edgecolor='black')
                ax.axvline(x=fwer, color='r', linestyle='dashed')
                ax.axvline(x=alpha, color='k', linestyle='dashed')
                if j == 0:
                    ax.set_ylabel(fr'$n = {n_}$')
                if i == 2:
                    ax.set_xlabel(fr'$\mu_1 = {mu1_}$')

                fdp = FDP(results['true'], results[algorithm])
                fdr = fdp.mean()
                ax = figs4[k][1][i][j]
                ax.hist(fdp, bins=50, range=(0, alpha * 2), density=True, edgecolor='black')
                ax.axvline(x=fdr, color='r', linestyle='dashed')
                ax.axvline(x=alpha, color='k', linestyle='dashed')
                if j == 0:
                    ax.set_ylabel(fr'$n = {n_}$')
                if i == 2:
                    ax.set_xlabel(fr'$\mu_1 = {mu1_}$')

    for k, algorithm in enumerate(['bonferroni', 'holm_bonferroni', 'hochberg', 'benjamini_hochberg']):
        fig = figs3[k][0]
        fig.suptitle(fr'Distribution of FWE for {algorithm}')
        fig.tight_layout()
        fig.savefig(figures_3[k])

        fig = figs4[k][0]
        fig.suptitle(fr'Distribution of FDP for {algorithm}')
        fig.tight_layout()
        fig.savefig(figures_4[k])
