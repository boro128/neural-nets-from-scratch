from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum_coeff: float = 0.0) -> None:
        super().__init__(lr)
        self._params = None

        assert 0 <= momentum_coeff < 1
        self._momentum_coeff = momentum_coeff

    def step(self) -> None:
        for param in self._params:
            param.momentum_weights = param.dweights + self._momentum_coeff * param.momentum_weights
            param.momentum_bias = param.dbias + self._momentum_coeff * param.momentum_bias
            param.weights -= self._lr * param.momentum_weights
            param.bias -= self._lr * param.momentum_bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self._lr}, momentum_coeff={self._momentum_coeff})"
