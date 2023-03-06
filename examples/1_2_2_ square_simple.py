from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/square-simple-training.csv", index_col=0)
df_test = pd.read_csv("datasets/regression/square-simple-test.csv", index_col=0)

X_train = df_train["x"].values.reshape((-1, 1))
y_train = df_train["y"].values.reshape((-1, 1))

X_test = df_test["x"].values.reshape((-1, 1))
y_test = df_test["y"].values.reshape((-1, 1))

scaler_X = mm.preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = mm.preprocessing.StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test_transf = scaler_y.transform(y_test)

# two hidden layers, 5 neurons each

model = mm.Model()

model.add(mm.LayerDense(1, 5))
model.add(mm.activation.Sigmoid())
model.add(mm.LayerDense(5, 5))
model.add(mm.activation.Sigmoid())
model.add(mm.LayerDense(5, 1))

model.set_optimizer(mm.optim.SGD())
model.set_loss(mm.loss.MSELoss())

model.train(X_train, y_train, n_epochs=1000, batch_size=1)

model.draw_weights()

out = model.forward(X_test)
out = scaler_y.reverse_transform(out)
mse = np.mean(np.square(out - y_test), axis=0).squeeze()
print(f"one hidden layer, 5 neurons, mse: {mse}")
