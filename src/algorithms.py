import numpy as np


def bonferroni(pvalues, alpha=0.05):
    m = pvalues.shape[1]
    pred = np.zeros_like(pvalues, dtype=int)
    pred[pvalues < alpha / m] = 1
    return pred

def holm_bonferroni(pvalues, alpha=0.05):
    reps, m = pvalues.shape
    pred = np.zeros_like(pvalues, dtype=int)
    ind = np.argsort(pvalues, axis=-1)
    pvalues_sorted = pvalues[np.arange(reps)[:, np.newaxis], ind]
    i = np.where(np.arange(m) < np.argmax(pvalues_sorted > alpha / (m - np.arange(m)), axis=1)[:, np.newaxis])
    pred[i[0], ind[i]] = 1
    return pred

def hochberg(pvalues, alpha=0.05):
    reps, m = pvalues.shape
    pred = np.zeros_like(pvalues, dtype=int)
    ind = np.argsort(pvalues, axis=-1)
    pvalues_sorted = pvalues[np.arange(reps)[:, np.newaxis], ind]
    i = pvalues.shape[1] - np.argmax((pvalues_sorted <= alpha / (m - np.arange(m)))[:, ::-1], axis=1) - 1 # last argmax in array
    i = np.where(np.arange(m) <= i[:, np.newaxis])
    pred[i[0], ind[i]] = 1
    return pred

def benjamini_hochberg(pvalues, alpha=0.05, independent=True):
    reps, m = pvalues.shape
    pred = np.zeros_like(pvalues, dtype=int)
    C = 1 if independent else np.sum(1 / np.arange(1, m + 1))
    ind = np.argsort(pvalues, axis=-1)
    pvalues_sorted = pvalues[np.arange(reps)[:, np.newaxis], ind]
    i = pvalues.shape[1] - np.argmax((pvalues_sorted < alpha * (np.arange(m) + 1) / (C * m))[:, ::-1], axis=1) - 1 # last argmax in array
    i = np.where(np.arange(m) <= i[:, np.newaxis])
    pred[i[0], ind[i]] = 1
    return pred
