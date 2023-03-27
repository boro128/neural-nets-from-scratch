from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._inputs = None
        self._output = None
        self._dinputs = None

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> None:
        pass

    @abstractmethod
    def backward(self, dvalues: np.ndarray) -> None:
        pass

    @property
    def output(self):
        return self._output

    @property
    def dinputs(self):
        return self._dinputs
