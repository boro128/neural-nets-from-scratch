import numpy as np


def softmax(inputs: np.ndarray) -> np.ndarray:
    exp = np.exp(inputs)
    return exp / np.sum(exp)


def fscore(inputs: np.ndarray, target: np.ndarray, label: int = 1) -> float:
    # Fscore = 2 * (precision * recall) / (precision + recall)

    tp = np.sum((target == label) & (inputs == label))
    fp = np.sum((target != label) & (inputs == label))
    fn = np.sum((inputs != label) & (target == label))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


def fscore_macro(inputs: np.ndarray, target: np.ndarray):
    return np.mean([fscore(inputs, target, label) for label in np.unique(target)])
