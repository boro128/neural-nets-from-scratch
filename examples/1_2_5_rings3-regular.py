from context import mm

import numpy as np
import pandas as pd
import itertools

df_train = pd.read_csv("datasets/classification/rings3-regular-training.csv")
df_test = pd.read_csv("datasets/classification/rings3-regular-test.csv")

X_train = df_train[["x", "y"]].values
y_train = pd.get_dummies(df_train["c"]).values

X_test = df_test[["x", "y"]].values
y_test = pd.get_dummies(df_test["c"]).values

scaler_X = mm.preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)


def execute(
    batch_size, n_epochs, optimizer, n_layers, n_neurons, activation, print_every=1000
):
    model = mm.Model()

    model.add(mm.LayerDense(2, n_neurons))

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

    model.add(mm.LayerDense(n_neurons, 3))

    model.set_optimizer(optimizer)
    model.set_loss(mm.loss.CrossEntropyLoss())

    model.init_weights_xavier()

    model.train(
        X_train,
        y_train,
        n_epochs=n_epochs,
        batch_size=batch_size,
        print_every=print_every,
    )

    out = model.forward(X_test)
    preds = np.argmax(out, axis=1)
    target = np.argmax(y_test, axis=1)
    f_score = mm.functional.fscore_macro(preds, target)
    print(
        f"softmax, {n_layers=}, {n_neurons=}, {activation=}, {batch_size=}, {n_epochs=}, {optimizer=}, f_score: {f_score}"
    )

    model.save(
        filename=f"1_2_5_rings3-regular__softmax__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


layers = [3]
neurons = [10]
activations = ["sigmoid", "tanh"]

for n_layers, n_neurons, activation in itertools.product(layers, neurons, activations):
    execute(
        batch_size=32,
        n_epochs=10_000,
        optimizer=mm.optim.SGD(lr=3e-3, momentum_coeff=0.3),
        print_every=2_000,
        n_layers=n_layers,
        n_neurons=n_neurons,
        activation=activation,
    )


###################### Output ######################

# epoch: 2000  loss: 0.9801805505714181
# epoch: 4000  loss: 0.9273174436023249
# epoch: 6000  loss: 0.9082114717110575
# epoch: 8000  loss: 0.9072778616157698
# epoch: 10000  loss: 0.9060168733570452
# softmax, n_layers=3, n_neurons=10, activation='sigmoid', batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.003, momentum_coeff=0.3), f_score: 0.567979697646411
# epoch: 2000  loss: 0.050941897961427604
# epoch: 4000  loss: 0.0395534883739566
# epoch: 6000  loss: 0.03277461617774563
# epoch: 8000  loss: 0.022456546293135836
# epoch: 10000  loss: 0.023806416974083526
# softmax, n_layers=3, n_neurons=10, activation='tanh', batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.003, momentum_coeff=0.3), f_score: 0.9681853754105806
