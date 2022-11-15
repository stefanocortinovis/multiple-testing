import numpy as np
from scipy.stats import norm


def get_means(mu0, mu1, m0, m1):
    return np.concatenate((np.ones(m0) * mu0, np.ones(m1) * mu1))

def get_t_statistics_z_test(m, n, reps, mus, mu0):
    return np.sqrt(n) * (np.random.randn(n, reps, m) + mus - mu0).mean(axis=0)

def get_pvalues_z_test(t_statistics):
    return 2 * norm.cdf(-abs(t_statistics))

def FWE(true, pred):
    false_reject = pred.copy()
    false_reject *= (1 - true)
    return np.any(false_reject, axis=-1).astype(int)

def FDP(true, pred):
    total_reject = pred.sum(axis=-1)
    total_reject[total_reject == 0] = 1
    false_reject = pred.copy()
    false_reject *= (1 - true)
    false_reject = false_reject.sum(axis=-1)
    return false_reject / total_reject
