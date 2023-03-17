import numpy as np

from .optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, lr: float = 0.01, beta: float = .0, eps: float = 1e-8) -> None:
        super().__init__(lr)
        self._beta = beta
        self._eps = eps

    def step(self) -> None:
        for param in self._params:
            param.v_weights = self._beta * param.v_weights + (1 - self._beta) * np.square(param.dweights)
            param.v_bias = self._beta * param.v_bias + (1 - self._beta) * np.square(param.dbias)
            param.weights -= self._lr * (param.dweights / (np.sqrt(param.v_weights) + self._eps))
            param.bias -= self._lr * (param.dbias / (np.sqrt(param.v_bias) + self._eps))
