import numpy as np

from .loss import Loss


class MSELoss(Loss):
    def calculate(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert len(inputs) == len(target)

        self._inputs = inputs
        self._target = target
        self._output = np.mean(np.square(inputs - target))
        return self._output

    def backward(self) -> None:
        n_instances = len(self._inputs)
        self._dinputs = 2 * (self._inputs - self._target) / n_instances
