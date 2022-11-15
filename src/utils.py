import numpy as np
from scipy.stats import norm


def get_means(mu0, mu1, m0, m1):
    return np.concatenate((np.ones(m0) * mu0, np.ones(m1) * mu1))

def get_t_statistics_z_test(m, n, reps, mus, mu0):
    return np.sqrt(n) * (np.random.randn(n, reps, m) + mus - mu0).mean(axis=0)

def get_pvalues_z_test(t_statistics):
    return 2 * norm.cdf(-abs(t_statistics))
