import numpy as np


class Sigmoid:
    def __init__(self):
        self._output = None

    def forward(self, inputs: np.ndarray) -> None:
        self._output = 1 / (1 + np.exp(-inputs))

    @property
    def output(self):
        return self._output
