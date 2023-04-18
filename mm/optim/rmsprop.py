import numpy as np

from .optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(
        self,
        lr: float = 0.01,
        beta: float = 0.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr)
        self._beta = beta
        self._eps = eps

        assert 0 <= weight_decay
        self._weight_decay = weight_decay

    def step(self) -> None:
        for param in self._params:
            param.dweights += self._weight_decay * param.weights
            param.dbias += self._weight_decay * param.bias

            param.v_weights = self._beta * param.v_weights + (1 - self._beta) * np.square(param.dweights)
            param.v_bias = self._beta * param.v_bias + (1 - self._beta) * np.square(param.dbias)

            param.weights -= self._lr * (param.dweights / (np.sqrt(param.v_weights) + self._eps))
            param.bias -= self._lr * (param.dbias / (np.sqrt(param.v_bias) + self._eps))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self._lr}, beta={self._beta}, eps={self._eps})"
