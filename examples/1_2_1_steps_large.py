from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/steps-large-training.csv", index_col=0)
df_test = pd.read_csv("datasets/regression/steps-large-test.csv", index_col=0)

X = df_test["x"].values.reshape((-1, 1))
y = df_test["y"].values

model = mm.MLP()

layer1 = mm.Layer_Dense(1, 5)
layer1.weights = np.array([[-27.626356, -20.89344, 27.120974, 45.69689, 11.612845]])
layer1.bias = np.array([[41.357834, -10.339673, 13.470915, -23.172596, 5.6102886]])

layer2 = mm.Layer_Dense(5, 5)
layer2.weights = np.array(
    [
        [-8.3284931e00, -4.4820315e-01, -3.0572968e01, 4.7598863e00, -3.6684063e01],
        [-1.5244807e00, 8.1501192e-01, -2.4369020e00, 2.3618015e01, -2.3995929e00],
        [-1.9370271e00, 6.7774868e-01, 4.2990065e00, -2.9325153e01, 4.3877707e00],
        [3.7552052e01, 3.2916479e00, 3.6218293e00, -3.6952205e00, 5.3370337e00],
        [-5.3912705e-01, 1.2272812e-02, 4.9252028e00, -1.2176819e01, 5.8789783e00],
    ]
)
layer2.bias = np.array([[-4.7462187, -0.85367996, 1.7170361, 2.4019506, 1.8899131]])

layer3 = mm.Layer_Dense(5, 1)
layer3.weights = np.array(
    [[58.631855], [37.152008], [35.596027], [-80.74089], [43.72223]]
)
layer3.bias = np.array([[-13.367186]])

model.add(layer1)
model.add(mm.activation.Sigmoid())
model.add(layer2)
model.add(mm.activation.Sigmoid())
model.add(layer3)

out = model.forward(X).squeeze()

mse = np.mean(np.square(out - y))
print(mse)
