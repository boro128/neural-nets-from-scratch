import numpy as np


class MSELoss:
    def __init__(self) -> None:
        self._dinputs = None
        self._target = None
        self._inputs = None
        self._output = None

    def calculate(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._inputs = inputs
        self._target = target
        self._output = np.mean(np.square(inputs - target))
        return self._output

    def backward(self) -> None:
        assert len(self._inputs) == len(self._target)

        n_instances = len(self._inputs)
        self._dinputs = 2 * (self._inputs - self._target) / n_instances

    @property
    def dinputs(self):
        return self._dinputs
    