import numpy as np

from .loss import Loss


class CrossEntropyLoss(Loss):
    def calculate(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert len(inputs) == len(target)

    def backward(self) -> None:
        pass
