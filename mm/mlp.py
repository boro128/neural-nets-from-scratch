from mm import LayerInput
import numpy as np


class MLP:
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
