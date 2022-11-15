from src.algorithms import bonferroni, holm_bonferroni, hochberg, benjamini_hochberg
from src.utils import get_means, get_t_statistics_z_test, get_pvalues_z_test


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
