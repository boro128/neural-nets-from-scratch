from mm import LayerInput, LayerDense
from mm.optim import SGD
from mm.loss import MSELoss

import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self) -> None:
        self._optimizer = None
        self._loss = None
        self._layers = [LayerInput()]
        self._trainable_layers = []
        self._rng = np.random.default_rng(seed=123)

    def add(self, layer) -> None:
        assert hasattr(layer, "forward")
        assert callable(layer.forward)

        self._layers.append(layer)

        if type(layer) is LayerDense:
            self._trainable_layers.append(layer)

    def set_loss(self, loss: MSELoss) -> None:
        self._loss = loss

    def set_optimizer(self, optimizer: SGD) -> None:
        self._optimizer = optimizer

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._layers[0].forward(inputs)

        prev = self._layers[0]
        for layer in self._layers[1:]:
            layer.forward(prev.output)
            prev = layer

        return self._layers[-1].output

    def train(
        self, X: np.ndarray, y: np.ndarray, n_epochs: int = 1, batch_size: int = None
    ):
        assert len(X) == len(y)
        n_instances = len(X)

        if batch_size is None:
            batch_size = len(X)

        self._optimizer.set_params(self._trainable_layers)

        for epoch_num in range(n_epochs):

            losses_epoch = []

            # split data into batches
            idxs = np.arange(n_instances)
            self._rng.shuffle(idxs)
            n_batches = n_instances // batch_size
            batches_idxs = np.array_split(idxs, n_batches)

            for batch_idxs in batches_idxs:
                X_batch = X[batch_idxs]
                y_batch = y[batch_idxs]

                output = self.forward(X_batch)
                loss_batch = self._loss.calculate(output, y_batch)
                losses_epoch.append(loss_batch)

                # backpropagation
                self._loss.backward()
                dvalues = self._loss.dinputs
                for layer in reversed(self._layers[1:]):
                    layer.backward(dvalues)
                    dvalues = layer.dinputs

                # parameters update
                self._optimizer.step()

            print(f"epoch: {epoch_num}  loss: {np.mean(losses_epoch)}")

    def draw_weights(self) -> None:
        layers_dense = []
        for layer in self._layers:
            if type(layer) is LayerDense:
                layers_dense.append(layer)

        fig, _ = plt.subplots(1, len(layers_dense))
        for idx, ax in enumerate(fig.axes):
            layers_dense[idx].visualize(
                ax, title=f"Layer {idx+1}", cbar=idx == len(fig.axes) - 1
            )
        fig.suptitle("Weights Visualization (last column is bias)")
        plt.show()

    @property
    def layers(self):
        return self._layers

    @property
    def trainable_layers(self):
        return self._trainable_layers
