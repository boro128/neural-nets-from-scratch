import numpy as np


def softmax(inputs: np.ndarray) -> np.ndarray:
    exp = np.exp(inputs)
    return exp / np.sum(exp)
