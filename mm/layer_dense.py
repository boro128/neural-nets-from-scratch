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
        self._momentum_weights = np.zeros_like(self._weights)
        self._momentum_bias = np.zeros_like(self._bias)
        self._v_weights = np.zeros_like(self._weights)
        self._v_bias = np.zeros_like(self._bias)

    def forward(self, inputs: np.ndarray) -> None:
        self._inputs = inputs
        self._output = inputs @ self._weights + self._bias

    def backward(self, dvalues: np.ndarray) -> None:
        self._dweights = self._inputs.T @ dvalues
        self._dbias = np.sum(dvalues, axis=0)
        self._dinputs = dvalues @ self._weights.T

    def init_weights_uniform(self, a=0, b=1, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng()
        self._weights = rng.uniform(a, b, (self._n_inputs, self._n_outputs))

    def init_weights_xavier(self, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng()
        a = np.sqrt(6) / np.sqrt(self._n_inputs + self._n_outputs)
        self._weights = rng.uniform(-a, a, (self._n_inputs, self._n_outputs))

    def visualize(self, ax=None, title=None, cbar=False) -> None:
        if ax is None:
            _, ax = plt.subplots()

        data = np.column_stack((self._weights.T, self._bias))
        sns.heatmap(data, ax=ax, cmap="seismic", vmin=-20, vmax=20, cbar=cbar)
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

    @property
    def momentum_weights(self):
        return self._momentum_weights

    @property
    def momentum_bias(self):
        return self._momentum_bias

    @property
    def v_weights(self):
        return self._v_weights

    @property
    def v_bias(self):
        return self._v_bias

    @weights.setter
    def weights(self, weights: np.ndarray):
        assert weights.shape == self._weights.shape
        self._weights = weights

    @bias.setter
    def bias(self, bias: np.ndarray):
        assert bias.shape == self._bias.shape
        self._bias = bias

    @dweights.setter
    def dweights(self, dweights: np.ndarray):
        assert dweights.shape == self._dweights.shape
        self._dweights = dweights

    @dbias.setter
    def dbias(self, dbias: np.ndarray):
        assert dbias.shape == self._dbias.shape
        self._dbias = dbias

    @momentum_weights.setter
    def momentum_weights(self, momentum_weights: np.ndarray):
        assert momentum_weights.shape == self._momentum_weights.shape
        self._momentum_weights = momentum_weights

    @momentum_bias.setter
    def momentum_bias(self, momentum_bias: np.ndarray):
        assert momentum_bias.shape == self._momentum_bias.shape
        self._momentum_bias = momentum_bias

    @v_weights.setter
    def v_weights(self, v_weights: np.ndarray):
        assert v_weights.shape == self._v_weights.shape
        self._v_weights = v_weights

    @v_bias.setter
    def v_bias(self, v_bias: np.ndarray):
        assert v_bias.shape == self._v_bias.shape
        self._v_bias = v_bias
