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


def execute(batch_size, n_epochs, print_every=1000, draw_weights=False):
    model = mm.Model()

    model.add(mm.LayerDense(1, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 1))

    model.set_optimizer(mm.optim.SGD())
    model.set_loss(mm.loss.MSELoss())

    model.init_weights_xavier()

    model.train(
        X_train,
        y_train,
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
# batch_size=1
execute(batch_size=1, n_epochs=3000, print_every=1000)

# two hidden layers, 5 neurons each
# batch_size=10
execute(batch_size=10, n_epochs=20_000, print_every=5000)

# two hidden layers, 5 neurons each
# batch_size=whole dataset
execute(batch_size=None, n_epochs=200_000, print_every=50_000)

# output;
# epoch: 1000  loss: 0.0007435554765755554
# epoch: 2000  loss: 0.0003916476501218075
# epoch: 3000  loss: 0.0002364691620495575
# batch_size=1, n_epochs=3000 mse: 2.2696451764901493
# epoch: 5000  loss: 0.0011185880847739214
# epoch: 10000  loss: 0.0006738650584537937
# epoch: 15000  loss: 0.00047418679088143877
# epoch: 20000  loss: 0.0003630403401623947
# batch_size=10, n_epochs=20000 mse: 4.02358382781651
# epoch: 50000  loss: 0.0011057558290373797
# epoch: 100000  loss: 0.0006602219401620602
# epoch: 150000  loss: 0.0004680576691515503
# epoch: 200000  loss: 0.0003554184532142622
# batch_size=None, n_epochs=200000 mse: 4.017415325970267
