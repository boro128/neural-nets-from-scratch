from .optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def step(self) -> None:
        pass
