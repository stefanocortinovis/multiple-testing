from functools import lru_cache, wraps

import numpy as np
from scipy.special import comb
from scipy.stats import norm


def get_means(mu0, mu1, m0, m1):
    return np.concatenate((np.ones(m0) * mu0, np.ones(m1) * mu1))


def get_t_statistics_z_test(m, n, reps, mus, mu0):
    return np.sqrt(n) * (np.random.randn(n, reps, m) + mus - mu0).mean(axis=0)


def get_pvalues_z_test(t_statistics):
    return 2 * norm.cdf(-abs(t_statistics))


def F0_unif(pvalues):
    return pvalues


def F1_z_test(pvalues, mu0, mu1, n=1):
    inv_pvalues = norm.ppf(pvalues / 2)
    mean_h1 = np.sqrt(n) * (mu1 - mu0)
    return 1 - norm.cdf(-inv_pvalues - mean_h1) + norm.cdf(inv_pvalues - mean_h1)


def G(pvalues, pi0, F0, F1):
    return pi0 * F0(pvalues) + (1 - pi0) * F1(pvalues)


def np_cache(*args, **kwargs):
    """ LRU cache implementation for functions whose FIRST parameter is a numpy array
        forked from: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75 """

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = tuple(np_array)
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    return decorator


@np_cache(maxsize=None)
def psi(t):
    """ Bolshev's recursion """
    k = len(t)
    if k == 0:
        return 1
    elif k == 1:
        return t[0]
    s = 1
    for i in range(1, k + 1):
        c = comb(k, i) * (1 - t[k - i]) ** i * psi(t[:(k - i)])
        s -= c
    return s.astype(float)


def D(t, m, k):
    return comb(m, k) * t[k - 1] ** k * psi(1 - t[k:m][::-1])  # 0 <= k <= m
