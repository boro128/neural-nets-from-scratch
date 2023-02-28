import numpy as np


class LayerInput:
    def __init__(self):
        self._output = None

    def forward(self, inputs: np.ndarray):
        self._output = inputs

    @property
    def output(self):
        return self._output
