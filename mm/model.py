from .layer_input import LayerInput
from .layer_dense import LayerDense
from mm.optim import Optimizer
from mm.loss import MSELoss

import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import pathlib
from datetime import datetime


class Model:
    def __init__(self) -> None:
        self._optimizer = None
        self._loss = None
        self._layers = [LayerInput()]
        self._trainable_layers = []
        self._rng = np.random.default_rng(seed=123)
        self._curr_epoch = 0

    def add(self, layer) -> None:
        assert hasattr(layer, "forward")
        assert callable(layer.forward)

        self._layers.append(layer)

        if type(layer) is LayerDense:
            self._trainable_layers.append(layer)

    def set_loss(self, loss: MSELoss) -> None:
        self._loss = loss

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._layers[0].forward(inputs)

        prev = self._layers[0]
        for layer in self._layers[1:]:
            layer.forward(prev.output)
            prev = layer

        return self._layers[-1].output

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_epochs: int = 1,
        batch_size: int = None,
        print_every: int = None,
        max_patience: int = 5,
    ) -> None:
        assert len(X) == len(y)
        n_instances = len(X)

        if batch_size is None:
            batch_size = len(X)

        self._optimizer.set_params(self._trainable_layers)
        val_losses = []
        patience = 0

        for _ in range(n_epochs):

            self._curr_epoch += 1
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

            # validation
            val_output = self.forward(X_val)
            val_loss = self._loss.calculate(val_output, y_val)
            val_losses.append(val_loss)
            if val_loss > val_losses[-1]:
                patience += 1
                if patience == max_patience:
                    print(f"Validation loss started to increase -- Early Stopping")
                    return
            else:
                patience = 0

            if print_every is not None and self._curr_epoch % print_every == 0:
                print(
                    f"epoch: {self._curr_epoch}  loss: {np.mean(losses_epoch)} val_loss: {val_loss}"
                )

    def init_weights_uniform(self, a=0, b=1) -> None:
        for layer in self._trainable_layers:
            layer.init_weights_uniform(a, b, self._rng)

    def init_weights_xavier(self) -> None:
        for layer in self._trainable_layers:
            layer.init_weights_xavier(self._rng)

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

    def save(self, directory: str = None, filename: str = None) -> None:
        model = copy.deepcopy(self)

        if directory is None:
            path = pathlib.Path.cwd().joinpath("models")
            path.mkdir(exist_ok=True)
        else:
            path = pathlib.Path(directory)

        if filename is None:
            filename = f'{datetime.now().strftime(r"%Y%m%d-%H%M%S")}.mmodel'

        path = path.joinpath(filename)

        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path) -> "Model":
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model

    @property
    def layers(self):
        return self._layers

    @property
    def trainable_layers(self):
        return self._trainable_layers
