import numpy as np


class LayerDense:
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self._weights = np.ones((n_inputs, n_outputs))
        self._bias = np.zeros(n_outputs)
        self._output = None

    def forward(self, inputs: np.ndarray) -> None:
        self._output = inputs @ self._weights + self._bias

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def output(self):
        return self._output

    @weights.setter
    def weights(self, weights: np.ndarray):
        assert weights.shape == self._weights.shape
        self._weights = weights

    @bias.setter
    def bias(self, bias: np.ndarray):
        assert bias.shape == self._bias.shape
        self._bias = bias
