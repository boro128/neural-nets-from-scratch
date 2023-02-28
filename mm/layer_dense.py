import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_outputs) -> None:
        self.weights = np.ones((n_inputs, n_outputs))
        self.bias = np.zeros((n_outputs))

    def forward(self, input) -> None:
        self.output = input @ self.weights + self.bias
