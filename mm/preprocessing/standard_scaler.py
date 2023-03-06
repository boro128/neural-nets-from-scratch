import numpy as np


class StandardScaler:
    def __init__(self) -> None:
        self._mean = None
        self._std = None

    def fit(self, inputs: np.ndarray) -> None:
        self._mean = np.mean(inputs, axis=0)
        self._std = np.std(inputs, axis=0)

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs - self._mean) / self._std

    def fit_transform(self, inputs: np.ndarray) -> np.ndarray:
        self.fit(inputs)
        return self.transform(inputs)

    def reverse_transform(self, inputs: np.ndarray) -> np.ndarray:
        return inputs * self._std + self._mean
