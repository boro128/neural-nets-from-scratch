from mm import Layer_Input


class MLP:
    def __init__(self) -> None:
        self.layers = [Layer_Input()]

    def add(self, layer) -> None:
        self.layers.append(layer)

    def forward(self, input) -> None:
        self.layers[0].forward(input)
        
        prev = self.layers[0]
        for layer in self.layers[1:]:
            layer.forward(prev.output)
            prev = layer

        return self.layers[-1].output
