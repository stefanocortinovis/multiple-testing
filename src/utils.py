import numpy as np
from scipy.stats import norm


def get_means(mu0, mu1, m0, m1):
    return np.concatenate((np.ones(m0) * mu0, np.ones(m1) * mu1))


def get_t_statistics_z_test(m, n, reps, mus, mu0):
    return np.sqrt(n) * (np.random.randn(n, reps, m) + mus - mu0).mean(axis=0)


def get_pvalues_z_test(t_statistics):
    return 2 * norm.cdf(-abs(t_statistics))


def cdf_pvalues_z_test_h1(pvalues, mu0, mu1, n=1):
    inv_pvalues = norm.ppf(pvalues / 2)
    mean_h1 = np.sqrt(n) * (mu1 - mu0)
    return 1 - norm.cdf(-inv_pvalues - mean_h1) + norm.cdf(inv_pvalues - mean_h1)


def FWE(true, pred):
    "Empirical family-wise error"
    false_reject = pred.copy()
    false_reject *= (1 - true)
    return np.any(false_reject, axis=-1).astype(int)


def FDP(true, pred):
    "Empirical family-wise error"
    total_reject = pred.sum(axis=-1)
    total_reject[total_reject == 0] = 1
    false_reject = pred.copy()
    false_reject *= (1 - true)
    false_reject = false_reject.sum(axis=-1)
    return false_reject / total_reject
