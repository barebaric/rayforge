"""Numerical gradient check shared by the constraint tests."""

import numpy as np


def check_grad(func, grad, x0, epsilon=1e-6):
    """
    Checks an analytical gradient against forward differences.

    Mirrors ``scipy.optimize.check_grad``: returns the L2 norm of the
    difference between the analytical and numerical gradients.
    """
    x0 = np.asarray(x0, dtype=float)
    analytical = np.asarray(grad(x0), dtype=float)
    numerical = np.empty_like(analytical)

    for index in range(x0.size):
        shifted = x0.copy()
        shifted[index] += epsilon
        numerical[index] = (func(shifted) - func(x0)) / epsilon

    return float(np.linalg.norm(analytical - numerical))
