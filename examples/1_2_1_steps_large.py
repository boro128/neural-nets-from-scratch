from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/steps-large-training.csv", index_col=0)
df_test = pd.read_csv("datasets/regression/steps-large-test.csv", index_col=0)

X = df_test["x"].values.reshape((-1, 1))
y = df_test["y"].values


# two hidden layers, 5 neurons each

model = mm.Model()

layer1 = mm.LayerDense(1, 5)
layer1.weights = np.array([[-213.6800, -92.7280, 168.5175, -177.1530, -226.7014]])
layer1.bias = np.array([-106.9620, 45.5648, -254.1388, 88.0355, -113.4121])

layer2 = mm.LayerDense(5, 5)
layer2.weights = np.array(
    [
        [-2.5420, -3.0759, -18.7080, 32.8780, -2.2470],
        [-2.7019, -11.7106, -7.8789, -8.2232, -8.5836],
        [40.1129, 7.0093, 6.7858, -3.6190, 42.5549],
        [-3.3904, -22.3619, -14.2373, -9.7289, -5.4421],
        [-2.6380, -2.1709, -14.0111, 34.8498, -1.8948],
    ]
)
layer2.bias = np.array([-29.9995, 10.8776, 8.8823, -14.7422, -7.5004])

layer3 = mm.LayerDense(5, 1)
layer3.weights = np.array([[32.6226], [41.4524], [38.7555], [-79.7586], [47.5424]])
layer3.bias = np.array([-0.0759])

model.add(layer1)
model.add(mm.activation.Sigmoid())
model.add(layer2)
model.add(mm.activation.Sigmoid())
model.add(layer3)

out = model.forward(X).squeeze()

mse = np.mean(np.square(out - y))
print(f"two hidden layers, 5 neurons each, mse: {mse}")

# one hidden layer, 10 neurons

model = mm.Model()

layer1 = mm.LayerDense(1, 10)
layer1.weights = np.array(
    [
        [
            104.78577,
            29.103115,
            50.641045,
            69.73865,
            75.232544,
            111.34952,
            71.25317,
            -112.89121,
            -113.42035,
            42.566055,
        ]
    ]
)
layer1.bias = np.array(
    [
        -52.28355,
        -43.72382,
        -76.04331,
        -34.773716,
        -112.92934,
        -55.56218,
        -106.96068,
        -56.31029,
        -56.574696,
        -21.233469,
    ]
)

layer2 = mm.LayerDense(10, 1)
layer2.weights = np.array(
    [
        [48.358549],
        [-43.259393],
        [14.0381965],
        [7.373916],
        [58.136595],
        [56.532244],
        [50.598963],
        [-39.926784],
        [-40.555776],
        [-32.40343],
    ]
)
layer2.bias = np.array([0.27252027])

model.add(layer1)
model.add(mm.activation.Sigmoid())
model.add(layer2)

out = model.forward(X).squeeze()

mse = np.mean(np.square(out - y))
print(f"one hidden layer, 10 neurons, mse: {mse}")

# one hidden layer, 5 neurons

model = mm.Model()

layer1 = mm.LayerDense(1, 5)
layer1.weights = np.array([[131.7981, 95.4253, 70.8011, 89.6914, 104.9499]])
layer1.bias = np.array([65.7796, -47.5908, -106.3184, -134.6479, -52.3460])

layer2 = mm.LayerDense(5, 1)
layer2.weights = np.array([[80.0541], [35.3729], [33.2226], [47.4418], [44.7309]])
layer2.bias = np.array([-80.1959])

model.add(layer1)
model.add(mm.activation.Sigmoid())
model.add(layer2)

out = model.forward(X).squeeze()

mse = np.mean(np.square(out - y))
print(f"one hidden layer, 5 neurons, mse: {mse}")

# output:
# two hidden layers, 5 neurons each, mse: 5.540121629860299
# one hidden layer, 10 neurons, mse: 15.720643735129217
# one hidden layer, 5 neurons, mse: 18.27054129800696
