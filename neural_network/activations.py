import numpy as np


def tanh(x) -> float:
    return np.tanh(x)


def tanh_derivative(x) -> float:
    return 1 - np.tanh(x) ** 2
