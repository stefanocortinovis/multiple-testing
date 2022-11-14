from src.algorithms import bonferroni, holm_bonferroni, hochberg, benjamini_hochberg
from src.utils import get_means, get_t_statistics_z_test, get_pvalues_z_test

import numpy as np


def z_test(mu0, mu1, m0, m1, n=1, alpha=0.05, algorithms='all'):
    """
    Known variance
    """
    if algorithms == 'all':
        algorithms = [bonferroni, holm_bonferroni, hochberg, benjamini_hochberg]

    m = m0 + m1
    mus = get_means(mu0, mu1, m0, m1)
    t_statistics = get_t_statistics_z_test(m, n, mus, mu0)
    pvalues = get_pvalues_z_test(t_statistics)

    results = {'true': (mus == mu0).astype(int)}
    for algorithm in algorithms:
        rejected = algorithm(pvalues, alpha)
        pred = np.zeros(m, dtype=int)
        pred[rejected] = 1
        results[algorithm.__name__] = pred
    
    return results
