from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/steps-small-training.csv", index_col=0)
df_test = pd.read_csv("datasets/regression/steps-small-test.csv", index_col=0)

X_train = df_train["x"].values.reshape((-1, 1))
y_train = df_train["y"].values.reshape((-1, 1))

X_test = df_test["x"].values.reshape((-1, 1))
y_test = df_test["y"].values.reshape((-1, 1))

scaler_X = mm.preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = mm.preprocessing.StandardScaler()
y_train_transf = scaler_y.fit_transform(y_train)
y_test_transf = scaler_y.transform(y_test)


def execute(batch_size, n_epochs, print_every=1000, draw_weights=False):
    model = mm.Model()

    model.add(mm.LayerDense(1, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 1))

    model.set_optimizer(mm.optim.SGD(0.1))
    model.set_loss(mm.loss.MSELoss())

    model.init_weights_xavier()

    model.train(
        X_train,
        y_train_transf,
        n_epochs=n_epochs,
        batch_size=batch_size,
        print_every=print_every,
    )

    out = model.forward(X_train)
    out = scaler_y.reverse_transform(out)
    mse = np.mean(np.square(out - y_train), axis=0).squeeze()
    print(f"{batch_size=}, {n_epochs=} mse_train: {mse}")

    out = model.forward(X_test)
    out = scaler_y.reverse_transform(out)
    mse = np.mean(np.square(out - y_test), axis=0).squeeze()
    print(f"{batch_size=}, {n_epochs=} mse_test: {mse}")

    if draw_weights:
        model.draw_weights()


# two hidden layers, 5 neurons each
# batch_size=10
execute(batch_size=10, n_epochs=50_000, print_every=5000)

# two hidden layers, 5 neurons each
# batch_size=whole dataset
# execute(batch_size=None, n_epochs=200_000, print_every=50_000)

# output:
# epoch: 5000  loss: 0.057904806131174644
# epoch: 10000  loss: 0.05600017670176162
# epoch: 15000  loss: 0.046986002017429584
# epoch: 20000  loss: 0.030616209378427954
# epoch: 25000  loss: 0.013093399456434526
# epoch: 30000  loss: 0.00789263781221618
# epoch: 35000  loss: 0.004140465772677798
# epoch: 40000  loss: 0.002751456301614933
# epoch: 45000  loss: 0.001980262989813
# epoch: 50000  loss: 0.0010989931905129487
# batch_size=10, n_epochs=50000 mse_train: 6.817633327489115
# batch_size=10, n_epochs=50000 mse_test: 107.50307323988395
