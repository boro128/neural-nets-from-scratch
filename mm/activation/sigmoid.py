import numpy as np


class Sigmoid:
    def forward(self, input: np.ndarray) -> None:
        self.output = 1 / (1 + np.exp(-input))
