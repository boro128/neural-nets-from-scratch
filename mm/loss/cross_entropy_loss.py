import numpy as np

from .loss import Loss


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        super().__init__()
        self._softmax_output = None

    def calculate(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert len(inputs) == len(target)

        self._inputs = inputs
        self._target = target

        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        self._softmax_output = probs_clipped

        correct_confidence = np.sum(probs_clipped * target, axis=1)

        self._output = np.mean(-np.log(correct_confidence))
        return self._output

    def backward(self) -> None:
        n_instances = len(self._inputs)
        self._dinputs = (self._softmax_output - self._target) / n_instances
