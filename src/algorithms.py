import numpy as np


def bonferroni(pvalues, alpha=0.05):
    m = len(pvalues)
    pred = np.zeros(m, dtype=int)
    pred[pvalues < alpha / m] = 1
    return pred

def holm_bonferroni(pvalues, alpha=0.05):
    m = len(pvalues)
    pred = np.zeros(m, dtype=int)
    ind = np.argsort(pvalues)
    pvalues_sorted = pvalues[ind]
    i = (pvalues_sorted > alpha / (m - np.arange(m))).nonzero()[0][0]
    pred[ind[:i]] = 1
    return pred

def hochberg(pvalues, alpha=0.05):
    m = len(pvalues)
    pred = np.zeros(m, dtype=int)
    ind = np.argsort(pvalues)
    pvalues_sorted = pvalues[ind]
    i = (pvalues_sorted <= alpha / (m - np.arange(m))).nonzero()[0][-1]
    pred[ind[:i+1]] = 1
    return pred

def benjamini_hochberg(pvalues, alpha=0.05, independent=True):
    m = len(pvalues)
    pred = np.zeros(m, dtype=int)
    C = 1 if independent else np.sum(1 / np.arange(1, m + 1))
    ind = np.argsort(pvalues)
    pvalues_sorted = pvalues[ind]
    i = (pvalues_sorted < alpha * (np.arange(m) + 1) / (C * m)).nonzero()[0][-1]
    pred[ind[:i+1]] = 1
    return pred
