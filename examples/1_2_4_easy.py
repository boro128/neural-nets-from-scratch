from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/classification/easy-training.csv")
df_test = pd.read_csv("datasets/classification/easy-test.csv")

X_train = df_train[["x", "y"]].values
y_train = pd.get_dummies(df_train["c"]).values

X_test = df_test[["x", "y"]].values
y_test = pd.get_dummies(df_test["c"]).values

scaler_X = mm.preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)


def execute_softmax(batch_size, n_epochs, optimizer, print_every=1000):
    model = mm.Model()

    model.add(mm.LayerDense(2, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 2))

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
    f_score = mm.functional.fscore(preds, target)
    print(f"softmax, {batch_size=}, {n_epochs=}, {optimizer=}, f_score: {f_score}")

    model.save(
        filename=f"1_2_4_easy__softmax__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


def execute_regular(batch_size, n_epochs, optimizer, print_every=1000):
    model = mm.Model()

    model.add(mm.LayerDense(2, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 1))

    model.set_optimizer(optimizer)
    model.set_loss(mm.loss.MSELoss())

    model.init_weights_xavier()

    y_train_local = np.argmax(y_train, axis=1, keepdims=True)
    y_test_local = np.argmax(y_test, axis=1)

    model.train(
        X_train,
        y_train_local,
        n_epochs=n_epochs,
        batch_size=batch_size,
        print_every=print_every,
    )

    out = model.forward(X_test)
    preds = np.where(out > 0.5, 1, 0).flatten()
    f_score = mm.functional.fscore(preds, y_test_local)
    print(f"regular, {batch_size=}, {n_epochs=}, {optimizer=}, f_score: {f_score}")

    model.save(
        filename=f"1_2_4_easy__regular__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


execute_softmax(32, 10_000, mm.optim.SGD(lr=3e-3, momentum_coeff=0.3), 2_000)
execute_regular(32, 10_000, mm.optim.SGD(lr=3e-3, momentum_coeff=0.3), 2_000)

###################### Experiment notes ######################

########### Softmax ###########

# softmax, batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.003, momentum_coeff=0.3), f_score: 0.9940119760479043

########### Regular ###########

# regular, batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.003, momentum_coeff=0.3), f_score: 0.9940119760479043
# regular, batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.003, momentum_coeff=0.3), f_score: 0.9900199600798402

###################### Output ######################

# epoch: 2000  loss: 0.06143390207298145
# epoch: 4000  loss: 0.03548568535301487
# epoch: 6000  loss: 0.026697684199889316
# epoch: 8000  loss: 0.0221421778943074
# epoch: 10000  loss: 0.019132244873526955
# softmax, batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.003, momentum_coeff=0.3), f_score: 0.9940119760479043
# epoch: 2000  loss: 0.04752699489624557
# epoch: 4000  loss: 0.03616269574083696
# epoch: 6000  loss: 0.030528508461053084
# epoch: 8000  loss: 0.02811290983623039
# epoch: 10000  loss: 0.02622855073966056
# regular, batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.003, momentum_coeff=0.3), f_score: 0.9900199600798402
