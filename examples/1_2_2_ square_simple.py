from context import mm
import numpy as np

layer = mm.LayerDense(1, 5)
layer.weights = np.array([[1, 2, 3, 100, 2]])

layer2 = mm.LayerDense(1, 5)
layer2.weights = np.array([[1, 2, 3, -100, 2]])
model = mm.MLP()

model.add(layer)
model.add(layer2)

model.draw_weights()
