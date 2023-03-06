import numpy as np


class MinMaxScaler:
    def __init__(self) -> None:
        self._min = None
        self._max = None

    def fit(self, inputs: np.ndarray) -> None:
        self._min = np.min(inputs, axis=0)
        self._max = np.max(inputs, axis=0)

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs - self._min) / (self._max - self._min)

    def fit_transform(self, inputs: np.ndarray) -> np.ndarray:
        self.fit(inputs)
        return self.transform(inputs)

    def reverse_transform(self, inputs: np.ndarray) -> np.ndarray:
        return inputs * (self._max - self._min) + self._min
