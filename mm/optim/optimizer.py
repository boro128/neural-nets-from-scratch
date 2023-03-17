from abc import ABC, abstractmethod
from typing import List

from mm import LayerDense


class Optimizer(ABC):
    def __init__(self, lr: float) -> None:
        super().__init__()
        self._params = None
        self._lr = lr

    def set_params(self, params: List[LayerDense]) -> None:
        self._params = params

    @abstractmethod
    def step(self) -> None:
        pass
