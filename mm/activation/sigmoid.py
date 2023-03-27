import numpy as np

from .activation import Activation


class Sigmoid(Activation):
    def forward(self, inputs: np.ndarray) -> None:
        self._inputs = inputs
        self._output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: np.ndarray) -> None:
        self._dinputs = dvalues * self._output * (1 - self._output)
