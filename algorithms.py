import numpy as np


def bonferroni(pvalues, alpha=0.05):
    m = len(pvalues)
    return np.arange(m)[pvalues < alpha / m]

def holm_bonferroni(pvalues, alpha=0.05):
    m = len(pvalues)
    ind = np.argsort(pvalues)
    pvalues_sorted = pvalues[ind]
    i = (pvalues_sorted > alpha / (m - np.arange(m))).nonzero()[0][0]
    return np.sort(ind[:i])

def hochberg(pvalues, alpha=0.05):
    m = len(pvalues)
    ind = np.argsort(pvalues)
    pvalues_sorted = pvalues[ind]
    i = (pvalues_sorted <= alpha / (m - np.arange(m))).nonzero()[0][-1]
    return np.sort(ind[:i+1])

def benjamini_hochberg(pvalues, alpha=0.05, independent=True):
    m = len(pvalues)
    C = 1 if independent else np.sum(1 / np.arange(1, m + 1))
    ind = np.argsort(pvalues)
    pvalues_sorted = pvalues[ind]
    i = (pvalues_sorted < alpha * (np.arange(m) + 1) / (C * m)).nonzero()[0][-1]
    return np.sort(ind[:i+1])
