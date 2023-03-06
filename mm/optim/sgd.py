import numpy as np
from typing import List

from mm import LayerDense


class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self._lr = lr

    def set_params(self, params: List[LayerDense]) -> None:
        self._params = params

    def step(self) -> None:
        for param in self._params:
            param.weights -= self._lr * param.dweights
            param.bias -= self._lr * param.dbias
