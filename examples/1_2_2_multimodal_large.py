from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/multimodal-large-training.csv")
df_test = pd.read_csv("datasets/regression/multimodal-large-test.csv")

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


def execute(batch_size, n_epochs, lr=0.01, print_every=1000, draw_weights=False):
    model = mm.Model()

    model.add(mm.LayerDense(1, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 1))

    model.set_optimizer(mm.optim.SGD(lr=lr))
    model.set_loss(mm.loss.MSELoss())

    model.init_weights_xavier()

    model.train(
        X_train,
        y_train_transf,
        n_epochs=n_epochs,
        batch_size=batch_size,
        print_every=print_every,
    )

    out = model.forward(X_test)
    out = scaler_y.reverse_transform(out)
    mse = np.mean(np.square(out - y_test), axis=0).squeeze()
    print(f"{batch_size=}, {n_epochs=} mse: {mse}")

    if draw_weights:
        model.draw_weights()


# two hidden layers, 5 neurons each
# batch_size=1000
execute(batch_size=1000, n_epochs=100_000, lr=0.1, print_every=25_000)

# output:
# epoch: 25000  loss: 0.061702192481562376
# epoch: 50000  loss: 0.059424186943216536
# epoch: 75000  loss: 0.010244721306949539
# epoch: 100000  loss: 0.002566187536929267
# batch_size=1000, n_epochs=100000 mse: 7.9521080325922595
