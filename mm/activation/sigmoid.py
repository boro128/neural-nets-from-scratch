import numpy as np


class Sigmoid:
    def forward(self, input) -> None:
        self.output = 1 / (1 + np.exp(-input))
