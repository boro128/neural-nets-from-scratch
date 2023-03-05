import numpy as np


class Sigmoid:
    def __init__(self):
        self._inputs = None
        self._output = None
        self._dinputs = None

    def forward(self, inputs: np.ndarray) -> None:
        self._inputs = inputs
        self._output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: np.ndarray) -> None:
        self._dinputs = dvalues * self._output * (1 - self._output)

    @property
    def output(self):
        return self._output
