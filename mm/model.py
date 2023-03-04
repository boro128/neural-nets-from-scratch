from mm import LayerInput, LayerDense
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self) -> None:
        self.layers = [LayerInput()]

    def add(self, layer) -> None:
        assert hasattr(layer, "forward")
        assert callable(layer.forward)

        self.layers.append(layer)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.layers[0].forward(inputs)

        prev = self.layers[0]
        for layer in self.layers[1:]:
            layer.forward(prev.output)
            prev = layer

        return self.layers[-1].output

    def draw_weights(self) -> None:
        layers_dense = []
        for layer in self.layers:
            if type(layer) is LayerDense:
                layers_dense.append(layer)

        fig, _ = plt.subplots(1, len(layers_dense))
        for idx, ax in enumerate(fig.axes):
            layers_dense[idx].visualize(
                ax, title=f"Layer {idx+1}", cbar=idx == len(fig.axes) - 1
            )
        fig.suptitle("Weights Visualization (last column is bias)")
        plt.show()
