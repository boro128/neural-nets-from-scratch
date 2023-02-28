from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/square-simple-training.csv", index_col=0)
df_test = pd.read_csv("datasets/regression/square-simple-test.csv", index_col=0)

X = df_test["x"].values.reshape((-1, 1))
y = df_test["y"].values

# two hidden layers, 5 neurons each

model = mm.MLP()

layer1 = mm.LayerDense(1, 5)
layer1.weights = np.array([[4.1357784, -4.966116, 6.4464865, -7.9883747, 5.592756]])
layer1.bias = np.array([-1.7032547, -1.7720956, -9.60833, 15.706666, 7.0029044])

layer2 = mm.LayerDense(5, 5)
layer2.weights = np.array(
    [
        [-5.223953, -7.603729, -12.939046, -4.4141293, -4.2426046],
        [-4.315546, -9.46816, -5.326138, 1.6418245, 1.119893],
        [-8.629373, -6.896841, -9.22218, -10.107484, -9.763134],
        [1.4223001, 6.7677526, 4.9257257, 12.381927, 14.164914],
        [6.4638653, 5.800526, 10.31827, 0.04026163, 0.64547027],
    ]
)
layer2.bias = np.array([-4.2617027, -2.02823, -2.5729602, 0.59431577, 1.9611947])

layer3 = mm.LayerDense(5, 1)
layer3.weights = np.array(
    [[-73.41892], [-77.798546], [-83.767465], [-50.89238], [-53.15942]]
)
layer3.bias = np.array([202.89932])

model.add(layer1)
model.add(mm.activation.Sigmoid())
model.add(layer2)
model.add(mm.activation.Sigmoid())
model.add(layer3)

out = model.forward(X).squeeze()

mse = np.mean(np.square(out - y))
print(f"two hidden layers, 5 neurons each, mse: {mse}")

# one hidden layer, 10 neurons

model = mm.MLP()

layer1 = mm.LayerDense(1, 10)
layer1.weights = np.array(
    [
        [
            -9.673959,
            -0.21904892,
            -11.088262,
            6.121494,
            11.075849,
            -9.836935,
            -7.7007814,
            14.651039,
            -9.578117,
            -9.422528,
        ]
    ]
)
layer1.bias = np.array(
    [
        9.307574,
        7.2209636,
        8.329594,
        4.2570133,
        -16.679883,
        4.2632492,
        -9.840837,
        -26.55505,
        11.508097,
        11.084183,
    ]
)

layer2 = mm.LayerDense(10, 1)
layer2.weights = np.array(
    [
        [-30.793924],
        [57.446616],
        [-32.344653],
        [-84.36993],
        [91.28587],
        [-30.9672420],
        [134.886172],
        [92.6416244],
        [-34.245793],
        [-33.797376],
    ]
)
layer2.bias = np.array([59.41757])

model.add(layer1)
model.add(mm.activation.Sigmoid())
model.add(layer2)

out = model.forward(X).squeeze()

mse = np.mean(np.square(out - y))
print(f"one hidden layer, 10 neurons, mse: {mse}")

# one hidden layer, 5 neurons

model = mm.MLP()

layer1 = mm.LayerDense(1, 5)
layer1.weights = np.array([[-8.54567, -11.664024, -13.445912, -19.833573, -8.429798]])
layer1.bias = np.array([-8.5419, 14.2041, 21.19816, 36.544794, 6.554697])

layer2 = mm.LayerDense(5, 1)
layer2.weights = np.array(
    [[158.760499], [-82.37453], [-94.95956], [-71.802744], [-79.4415]]
)
layer2.bias = np.array([213.4618])

model.add(layer1)
model.add(mm.activation.Sigmoid())
model.add(layer2)

out = model.forward(X).squeeze()

mse = np.mean(np.square(out - y))
print(f"one hidden layer, 5 neurons, mse: {mse}")
