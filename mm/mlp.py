from mm import Layer_Input
import numpy as np

class MLP:
    def __init__(self) -> None:
        self.layers = [Layer_Input()]

    def add(self, layer) -> None:
        assert hasattr(layer, "forward")
        assert callable(layer.forward)

        self.layers.append(layer)

    def forward(self, input: np.ndarray) -> None:
        self.layers[0].forward(input)

        prev = self.layers[0]
        for layer in self.layers[1:]:
            layer.forward(prev.output)
            prev = layer

        return self.layers[-1].output
