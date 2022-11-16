from src.hypotheses import z_test, eFDP, eFWE, eTPP, FDR_bh, POW_bh_z_test, F_FDP_hp_z_test
from src.utils import get_pvalues_z_test, get_t_statistics_z_test, F0_unif, F1_z_test

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)

mpl.rcParams.update({
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
    'axes.labelsize': 16,
})

figure_1 = './figures/pvalues_h0.png'
figure_2 = './figures/pvalues_h1.png'
figures_3 = ['./figures/FWER_bonferroni.png', './figures/FWER_holm_bonferroni.png', './figures/FWER_hochberg.png', './figures/FWER_benjamini_hochberg.png']
figures_4 = ['./figures/FDR_bonferroni.png', './figures/FDR_holm_bonferroni.png', './figures/FDR_hochberg.png', './figures/FDR_benjamini_hochberg.png']
figures_5 = ['./figures/FDR_bonferroni_cumulative.png', './figures/FDR_holm_bonferroni_cumulative.png', './figures/FDR_hochberg_cumulative.png', './figures/FDR_benjamini_hochberg_cumulative.png']
figures_6 = ['./figures/POW_bonferroni.png', './figures/POW_holm_bonferroni.png', './figures/POW_hochberg.png', './figures/POW_benjamini_hochberg.png']

if not os.path.isfile(figure_1):
    m0 = 10000
    mu0 = 0
    mu0_ = np.ones(m0) * mu0
    interval = np.linspace(0, 1, 1001)
    true_dist = F0_unif(interval)  # same as interval

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    t_statistics = get_t_statistics_z_test(m0, 1, 1, mu0_, mu0)
    pvalues = get_pvalues_z_test(t_statistics).squeeze()
    ax.plot(interval, true_dist, color='red', label='true CDF', linewidth=1.5, linestyle='dashed')
    ax.hist(pvalues, bins=50, range=(0, 1), density=True, cumulative=True, edgecolor='black', label='empirical CDF')
    ax.set_xlabel('p')
    ax.set_ylabel('density')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    fig.suptitle('CDF of p-values under $H_0$')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(figure_1)

if not os.path.isfile(figure_2):
    m1 = 10000
    mu0 = 0
    mu1 = [1, 1.5, 2.5]
    n = [1, 5, 15]
    interval = np.linspace(0, 1, 1001)

    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for i, n_ in enumerate(n):
        for j, mu1_ in enumerate(mu1):
            true_dist = F1_z_test(interval, mu0, mu1_, n_)
            mu1s = np.ones(m1) * mu1_
            t_statistics = get_t_statistics_z_test(m1, n_, 1, mu1s, mu0)
            pvalues = get_pvalues_z_test(t_statistics).squeeze()
            ax[i][j].plot(interval, true_dist, color='red', label='true CDF', linewidth=2.5, linestyle='dashed')
            ax[i][j].hist(pvalues, bins=50, range=(0, 1), density=True, cumulative=True, edgecolor='black', label='empirical CDF')
            ax[i][j].set_xlim(0, 1)
            ax[i][j].set_ylim(0, 1.1)
            if j == 0:
                ax[i][j].set_ylabel(fr'$n = {n_}$')
            if i == 2:
                ax[i][j].set_xlabel(fr'$\mu_1 = {mu1_}$')
    fig.legend(*ax[0][0].get_legend_handles_labels())
    fig.suptitle('CDF of p-values under $H_1$')
    plt.tight_layout()
    plt.savefig(figure_2)

if not os.path.isfile(figures_3[0]) or not os.path.isfile(figures_4[0]):
    alpha = 0.05
    m = 1000
    pi0 = 0.5
    mu0 = 0
    mu1 = [1, 1.5, 2.5]
    n = [1, 5, 15]
    reps = 10000

    figs3 = [plt.subplots(3, 3, figsize=(12, 12)) for i in range(4)]
    figs4 = [plt.subplots(3, 3, figsize=(12, 12)) for i in range(4)]
    for i, n_ in enumerate(n):
        for j, mu1_ in enumerate(mu1):
            results = z_test(m, pi0, mu0, mu1_, n=n_, reps=reps, alpha=alpha)
            for k, algorithm in enumerate(['bonferroni', 'holm_bonferroni', 'hochberg', 'benjamini_hochberg']):
                efwe = eFWE(results['true'], results[algorithm])
                efwer = efwe.mean()
                ax = figs3[k][1][i][j]
                ax.hist(efwe, bins=2, range=(0, 1), density=True, edgecolor='black')
                ax.axvline(x=efwer, color='r', linestyle='dashed')
                ax.axvline(x=alpha, color='k', linestyle='dashed')
                if j == 0:
                    ax.set_ylabel(fr'$n = {n_}$')
                if i == 2:
                    ax.set_xlabel(fr'$\mu_1 = {mu1_}$')

                efdp = eFDP(results['true'], results[algorithm])
                efdr = efdp.mean()
                ax = figs4[k][1][i][j]
                ax.hist(efdp, bins=50, range=(0, alpha * 2), density=True, edgecolor='black')
                ax.axvline(x=efdr, color='r', linestyle='dashed')
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
        if k == 3:
            fig.suptitle(fr'Distribution of FDP for {algorithm} with theoretical FDR = {FDR_bh(pi0, alpha):.3f}')
            fig.legend(*figs4[k][1][0][0].get_legend_handles_labels())
        else:
            fig.suptitle(fr'Distribution of FDP for {algorithm}')
        fig.tight_layout()
        fig.savefig(figures_4[k])

if not os.path.isfile(figures_5[0]) or not os.path.isfile(figures_6[0]):
    alpha = 0.05
    m = 50
    pi0 = 0.5
    mu0 = 0
    mu1 = [1, 1.5, 2.5]
    n = [1, 5, 15]
    reps = 10000
    interval = np.concatenate((np.linspace(0, alpha * 6, 21), np.linspace(alpha * 6, 1, 6)[1:]))

    figs5 = [plt.subplots(3, 3, figsize=(12, 12)) for i in range(4)]
    figs6 = [plt.subplots(3, 3, figsize=(12, 12)) for i in range(4)]
    for i, n_ in enumerate(n):
        for j, mu1_ in enumerate(mu1):
            results = z_test(m, pi0, mu0, mu1_, n=n_, reps=reps, alpha=alpha)
            for k, algorithm in enumerate(['bonferroni', 'holm_bonferroni', 'hochberg', 'benjamini_hochberg']):
                efdp = eFDP(results['true'], results[algorithm])
                efdr = efdp.mean()
                ax = figs5[k][1][i][j]
                ax.hist(efdp, bins=50, range=(0, 1), density=True, cumulative=True, edgecolor='black', label='empirical CDF')
                if j == 0:
                    ax.set_ylabel(fr'$n = {n_}$')
                if i == 2:
                    ax.set_xlabel(fr'$\mu_1 = {mu1_}$')
                if k == 3:
                    true_dist = [F_FDP_hp_z_test(x, alpha, m, pi0, mu0, mu1_, n_) for x in interval]
                    ax.plot(interval, true_dist, color='red', label='true CDF', linewidth=2.5, linestyle='dashed')

                etpp = eTPP(results['true'], results[algorithm])
                epow = etpp.mean()
                ax = figs6[k][1][i][j]
                ax.hist(etpp, bins=25, range=(0, 1), density=True, edgecolor='black')
                ax.axvline(x=epow, color='r', linestyle='dashed')
                if j == 0:
                    ax.set_ylabel(fr'$n = {n_}$')
                if i == 2:
                    ax.set_xlabel(fr'$\mu_1 = {mu1_}$')

                    tpow = POW_bh_z_test(alpha, m, pi0, mu0, mu1_, n_)
                    textstr = fr'Theoretical $POW = {tpow:.2f}$'
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    for k, algorithm in enumerate(['bonferroni', 'holm_bonferroni', 'hochberg', 'benjamini_hochberg']):
        fig = figs5[k][0]
        if k == 3:
            fig.suptitle(fr'Distribution of FDP for {algorithm} with theoretical FDR = {FDR_bh(pi0, alpha):.3f}')
            ax = figs5[k][1][0][2]
            ax.legend(*ax.get_legend_handles_labels(), loc='center')
        else:
            fig.suptitle(fr'Distribution of FDP for {algorithm}')
        fig.tight_layout()
        fig.savefig(figures_5[k])

        fig = figs6[k][0]
        fig.suptitle(fr'Distribution of TPP for {algorithm}')
        fig.tight_layout()
        fig.savefig(figures_6[k])
