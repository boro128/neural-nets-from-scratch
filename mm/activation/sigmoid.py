import numpy as np


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray) -> None:
        self.output = 1 / (1 + np.exp(-inputs))
