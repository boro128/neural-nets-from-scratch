import numpy as np

from .activation import Activation


class Tanh(Activation):
    def forward(self, inputs: np.ndarray) -> None:
        self._inputs = inputs
        exp_x = np.exp(inputs)
        exp_minus_x = np.exp(-inputs)
        self._output = (exp_x - exp_minus_x) / (exp_x + exp_minus_x)

    def backward(self, dvalues: np.ndarray) -> None:
        self._dinputs = dvalues * (1 - np.square(self._output))
