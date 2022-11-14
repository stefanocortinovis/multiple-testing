from algorithms import bonferroni, holm_bonferroni, hochberg, benjamini_hochberg

import numpy as np


alpha = 0.05
pvalues = np.array([
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
])
pvalues_sorted = np.sort(pvalues)

def test_bonferroni():
    assert np.all(bonferroni(pvalues, alpha) == np.array([1, 3]))
    assert np.all(bonferroni(pvalues_sorted, alpha) == np.array([0, 1]))

def test_holm_bonferroni():
    assert np.all(holm_bonferroni(pvalues, alpha) == np.array([1, 3, 5]))
    assert np.all(holm_bonferroni(pvalues_sorted, alpha) == np.array([0, 1, 2]))

def test_hochberg():
    assert np.all(hochberg(pvalues, alpha) == np.array([1, 2, 3, 5, 7, 8]))
    assert np.all(hochberg(pvalues_sorted, alpha) == np.array([0, 1, 2, 3, 4, 5]))

def test_benjamini_hochberg():
    assert np.all(benjamini_hochberg(pvalues, alpha) == np.array([1, 2, 3, 4, 5, 7, 8, 9]))
    assert np.all(benjamini_hochberg(pvalues_sorted, alpha) == np.array([0, 1, 2, 3, 4, 5, 6, 7]))
