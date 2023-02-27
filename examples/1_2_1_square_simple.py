from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/square-simple-training.csv", index_col=0)
df_test = pd.read_csv("datasets/regression/square-simple-test.csv", index_col=0)

X = df_test["x"].values.reshape((-1, 1))
y = df_test["y"].values

model = mm.MLP()

layer1 = mm.Layer_Dense(1, 5)
layer1.weights = np.array([[4.1357784, -4.966116, 6.4464865, -7.9883747, 5.592756]])
layer1.bias = np.array([[-1.7032547, -1.7720956, -9.60833, 15.706666, 7.0029044]])

layer2 = mm.Layer_Dense(5, 5)
layer2.weights = np.array(
    [
        [-5.2239532, -7.603729, -12.939046, -4.4141293, -4.2426047],
        [-4.315536, -9.46816, -5.326138, 1.6418235, 1.119893],
        [-8.629273, -6.896841, -9.22218, -10.107484, -9.763134],
        [1.4223001, 6.7677526, 4.9157257, 12.381927, 14.164914],
        [6.4638653, 5.800516, 10.31827, 0.04026163, 0.64547026],
    ]
)
layer2.bias = np.array([[-4.2617025, -2.02823, -2.5729601, 0.59431577, 1.9611949]])

layer3 = mm.Layer_Dense(5, 1)
layer3.weights = np.array(
    [[-73.4189], [-77.798546], [-83.767365], [-50.89138], [-53.15941]]
)
layer3.bias = np.array([[202.8993]])

model.add(layer1)
model.add(mm.activation.Sigmoid())
model.add(layer2)
model.add(mm.activation.Sigmoid())
model.add(layer3)

out = model.forward(X).squeeze()

mse = np.mean(np.square(out - y))
print(mse)
