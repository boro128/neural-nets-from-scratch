from context import mm

import numpy as np
import pandas as pd
import itertools

df_train = pd.read_csv("datasets/regression/steps-large-training.csv")
df_test = pd.read_csv("datasets/regression/steps-large-test.csv")

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


def execute(
    batch_size, n_epochs, optimizer, n_layers, n_neurons, activation, print_every=1000
):
    model = mm.Model()

    model.add(mm.LayerDense(1, n_neurons))

    for _ in range(n_layers):
        if activation == "sigmoid":
            model.add(mm.activation.Sigmoid())
        elif activation == "relu":
            model.add(mm.activation.ReLu())
        elif activation == "tanh":
            model.add(mm.activation.Tanh())
        model.add(mm.LayerDense(n_neurons, n_neurons))

    if activation == "sigmoid":
        model.add(mm.activation.Sigmoid())
    elif activation == "relu":
        model.add(mm.activation.ReLu())
    elif activation == "tanh":
        model.add(mm.activation.Tanh())

    model.add(mm.LayerDense(n_neurons, 1))

    model.set_optimizer(optimizer)
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
    print(
        f"{n_layers=}, {n_neurons=}, {activation=}, {batch_size=}, {n_epochs=}, {optimizer=}, mse: {mse}"
    )

    model.save(
        filename=f"1_2_5_steps_large__n_layers_{n_layers}__n_neurons_{n_neurons}__{activation}__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


layers = [3]
neurons = [10]
activations = ["sigmoid", "tanh"]

for n_layers, n_neurons, activation in itertools.product(layers, neurons, activations):
    execute(
        batch_size=256,
        n_epochs=40_000,
        optimizer=mm.optim.SGD(lr=3e-2, momentum_coeff=0.3),
        print_every=8_000,
        n_layers=n_layers,
        n_neurons=n_neurons,
        activation=activation,
    )

###################### Output ######################

# epoch: 8000  loss: 0.062270926112620875
# epoch: 16000  loss: 0.007527191348621971
# epoch: 24000  loss: 0.0042730506306329205
# epoch: 32000  loss: 0.003170033330769226
# epoch: 40000  loss: 0.0025905846177180216
# n_layers=3, n_neurons=10, activation='sigmoid', batch_size=256, n_epochs=40000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), mse: 14.428367815861122
# epoch: 8000  loss: 0.0013134428445009312
# epoch: 16000  loss: 0.0010920671614582543
# epoch: 24000  loss: 0.000672572098677818
# epoch: 32000  loss: 0.0008579883017468962
# epoch: 40000  loss: 0.0009064515224025388
# n_layers=3, n_neurons=10, activation='tanh', batch_size=256, n_epochs=40000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), mse: 2.059062617333513
