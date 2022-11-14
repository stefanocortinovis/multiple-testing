from src.algorithms import bonferroni, holm_bonferroni, hochberg, benjamini_hochberg

import numpy as np
from scipy.stats import norm


def z_test(mu0, mu1, m0, m1, n=1, alpha=0.05, algorithms='all'):
    """
    Known variance
    """
    if algorithms == 'all':
        algorithms = [bonferroni, holm_bonferroni, hochberg, benjamini_hochberg]

    m = m0 + m1
    mus = np.concatenate((np.ones(m0) * mu0, np.ones(m1) * mu1))
    test_statistics = np.sqrt(n) * (np.random.randn(m, n) + mus[:, np.newaxis] - mu0).mean(axis=1)
    pvalues = 2 * norm.cdf(-abs(test_statistics))

    results = {'true': (mus == mu0).astype(int)}
    for algorithm in algorithms:
        rejected = algorithm(pvalues, alpha)
        pred = np.zeros(m, dtype=int)
        pred[rejected] = 1
        results[algorithm.__name__] = pred
    
    return results
