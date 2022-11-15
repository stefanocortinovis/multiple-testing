from src.algorithms import bonferroni, holm_bonferroni, hochberg, benjamini_hochberg

import numpy as np


alpha = 0.05
pvalues = np.array([[
    0.1055,
    0.0045,
    0.0090,
    0.0020,
    0.0250,
    0.0060,
    0.5350,
    0.0085,
    0.0080,
    0.0175
]])
pvalues_sorted = np.sort(pvalues)
pvalues_stacked = np.vstack((pvalues, pvalues_sorted))

def test_bonferroni():
    assert np.all(bonferroni(pvalues, alpha) == np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0]]))
    assert np.all(bonferroni(pvalues_sorted, alpha) == np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]))
    assert np.all(bonferroni(pvalues_stacked, alpha) == np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]))

def test_holm_bonferroni():
    assert np.all(holm_bonferroni(pvalues, alpha) == np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0]]))
    assert np.all(holm_bonferroni(pvalues_sorted, alpha) == np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]))
    assert np.all(holm_bonferroni(pvalues_stacked, alpha) == np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]))

def test_hochberg():
    assert np.all(hochberg(pvalues, alpha) == np.array([[0, 1, 1, 1, 0, 1, 0, 1, 1, 0]]))
    assert np.all(hochberg(pvalues_sorted, alpha) == np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]))
    assert np.all(hochberg(pvalues_stacked, alpha) == np.array([[0, 1, 1, 1, 0, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]))

def test_benjamini_hochberg():
    assert np.all(benjamini_hochberg(pvalues, alpha) == np.array([[0, 1, 1, 1, 1, 1, 0, 1, 1, 1]]))
    assert np.all(benjamini_hochberg(pvalues_sorted, alpha) == np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]))
    assert np.all(benjamini_hochberg(pvalues_stacked, alpha) == np.array([[0, 1, 1, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]))
