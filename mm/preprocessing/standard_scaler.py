import numpy as np


class StandardScaler:
    def __init__(self) -> None:
        self._mean = None
        self._std = None

    def fit(self, inputs: np.ndarray) -> None:
        self._mean = np.mean(inputs)
        self._std = np.std(inputs)

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs - self._mean) / self._std

    def fit_transform(self, inputs: np.ndarray) -> np.ndarray:
        self.fit(inputs)
        return self.transform(inputs)
