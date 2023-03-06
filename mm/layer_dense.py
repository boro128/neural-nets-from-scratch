import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LayerDense:
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._weights = np.ones((n_inputs, n_outputs))
        self._bias = np.zeros(n_outputs)
        self._output = None
        self._inputs = None
        self._dweights = None
        self._dbias = None
        self._dinputs = None

    def forward(self, inputs: np.ndarray) -> None:
        self._inputs = inputs
        self._output = inputs @ self._weights + self._bias

    def backward(self, dvalues: np.ndarray) -> None:
        self._dweights = self._inputs.T @ dvalues
        self._dbias = np.sum(dvalues, axis=0)
        self._dinputs = dvalues @ self._weights.T

    # TODO
    def init_weights_uniform(self, a=0, b=1):
        self._weights = np.random.uniform(a, b, (self._n_inputs, self._n_outputs))

    def init_weights_xavier(self):
        a = np.sqrt(6) / np.sqrt(self._n_inputs + self._n_outputs)
        self._weights = np.random.uniform(-a, a, (self._n_inputs, self._n_outputs))

    def visualize(self, ax=None, title=None, cbar=False) -> None:
        if ax is None:
            _, ax = plt.subplots()

        data = np.column_stack((self._weights.T, self._bias))
        sns.heatmap(data, ax=ax, cmap="seismic", vmin=-100, vmax=100, cbar=cbar)
        ax.set_title(title)

        if ax is None:
            plt.show()

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._n_outputs

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def output(self):
        return self._output

    @property
    def dweights(self):
        return self._dweights

    @property
    def dbias(self):
        return self._dbias

    @property
    def dinputs(self):
        return self._dinputs

    @weights.setter
    def weights(self, weights: np.ndarray):
        assert weights.shape == self._weights.shape
        self._weights = weights

    @bias.setter
    def bias(self, bias: np.ndarray):
        assert bias.shape == self._bias.shape
        self._bias = bias
