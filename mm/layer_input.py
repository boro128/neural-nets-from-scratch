import numpy as np


class Layer_Input:
    def forward(self, input: np.ndarray):
        self.output = input
