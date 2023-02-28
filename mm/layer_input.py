import numpy as np


class LayerInput:
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray):
        self.output = inputs
