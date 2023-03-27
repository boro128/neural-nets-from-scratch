from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._dinputs = None
        self._target = None
        self._inputs = None
        self._output = None

    @abstractmethod
    def calculate(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self) -> None:
        pass

    @property
    def dinputs(self):
        return self._dinputs
