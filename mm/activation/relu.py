import numpy as np

from .activation import Activation


class ReLu(Activation):
    def forward(self, inputs: np.ndarray) -> None:
        self._inputs = inputs
        self._output = np.maximum(inputs, 0)

    def backward(self, dvalues: np.ndarray) -> None:
        self._dinputs = dvalues * np.where(self._dinputs <= 0, 0, self._dinputs)
