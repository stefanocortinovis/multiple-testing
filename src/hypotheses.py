from src.algorithms import bonferroni, holm_bonferroni, hochberg, benjamini_hochberg
from src.utils import get_means, get_t_statistics_z_test, get_pvalues_z_test, G, F0_unif, F1_z_test, psi

import numpy as np
from scipy.special import comb


def z_test(m, pi0, mu0, mu1, n=1, reps=1, alpha=0.05, algorithms='all'):
    """
    Known variance
    """
    if algorithms == 'all':
        algorithms = [bonferroni, holm_bonferroni, hochberg, benjamini_hochberg]

    m0 = int(pi0 * m)
    m1 = m - m0
    mus = get_means(mu0, mu1, m0, m1)
    t_statistics = get_t_statistics_z_test(m, n, reps, mus, mu0)
    pvalues = get_pvalues_z_test(t_statistics)

    results = {'true': (mus != mu0).astype(int)}
    for algorithm in algorithms:
        pred = algorithm(pvalues, alpha)
        results[algorithm.__name__] = pred

    return results


def eFWE(true, pred):
    "Empirical family-wise error"
    false_reject = pred.copy()
    false_reject *= (1 - true)
    return np.any(false_reject, axis=-1).astype(int)


def eFDP(true, pred):
    "Empirical false-discovery proportion"
    total_reject = pred.sum(axis=-1)
    total_reject[total_reject == 0] = 1
    false_reject = pred.copy()
    false_reject *= (1 - true)
    false_reject = false_reject.sum(axis=-1)
    return false_reject / total_reject


def eTPP(true, pred):
    "Empirical true positive proportion"
    reject = true.sum(axis=-1)
    true_reject = pred.copy()
    true_reject *= true
    true_reject = true_reject.sum(axis=-1)
    return true_reject / reject


def FDR_bh(pi0=0.5, alpha=0.05):
    return pi0 * alpha


def POW_bh_z_test(alpha, m, pi0, mu0, mu1, n):
    def F1(t):
        return F1_z_test(t, mu0, mu1, n)

    def G_(t):
        return G(t, pi0, F0_unif, F1)

    s = 0
    for k in range(1, m + 1):
        s += F1(alpha * k / m) * comb(m - 1, k - 1) * G_(alpha * k / m) ** (k - 1) * psi((1 - G_(alpha * np.arange(m, k, -1) / m)))
    return s
