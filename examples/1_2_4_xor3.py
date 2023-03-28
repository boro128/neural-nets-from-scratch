from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/classification/xor3-training.csv")
df_test = pd.read_csv("datasets/classification/xor3-test.csv")

X_train = df_train[["x", "y"]].values
y_train = pd.get_dummies(df_train["c"]).values

X_test = df_test[["x", "y"]].values
y_test = pd.get_dummies(df_test["c"]).values

scaler_X = mm.preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)


def execute_softmax(batch_size, n_epochs, optimizer, print_every=1000):
    model = mm.Model()

    model.add(mm.LayerDense(2, 10))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(10, 5))
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
        filename=f"1_2_4_xor3__softmax__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


def execute_regular(batch_size, n_epochs, optimizer, print_every=1000):
    model = mm.Model()

    model.add(mm.LayerDense(2, 10))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(10, 5))
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
        filename=f"1_2_4_xor3__regular__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


execute_softmax(32, 30_000, mm.optim.SGD(lr=3e-2, momentum_coeff=0.3), 6_000)
execute_regular(32, 30_000, mm.optim.SGD(lr=3e-2, momentum_coeff=0.3), 6_000)

###################### Experiment notes ######################

########### Softmax ###########

# softmax, batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), f_score: 0.9622641509433962
# softmax, batch_size=32, n_epochs=60000, optimizer=SGD(lr=0.03, momentum_coeff=0.35), f_score: 0.9622641509433962
# softmax, batch_size=32, n_epochs=40000, optimizer=SGD(lr=0.03, momentum_coeff=0.35), f_score: 0.9647058823529411
# softmax, batch_size=16, n_epochs=20000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), f_score: 0.9573459715639812
# softmax, batch_size=32, n_epochs=100000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), f_score: 0.9645390070921986
# softmax, batch_size=32, n_epochs=50000, optimizer=SGD(lr=0.007, momentum_coeff=0.3), f_score: 0.9691211401425179
# softmax, batch_size=32, n_epochs=100000, optimizer=SGD(lr=0.007, momentum_coeff=0.3), f_score: 0.9622641509433962

########### Regular ###########

# regular, batch_size=32, n_epochs=25000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), f_score: 0.9530516431924884
# regular, batch_size=32, n_epochs=100000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), f_score: 0.9534883720930233
# regular, batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), f_score: 0.9530516431924884

###################### Output ######################

# epoch: 6000  loss: 0.11501367456402058
# epoch: 12000  loss: 0.017609387126175467
# epoch: 18000  loss: 0.008900053867193494
# epoch: 24000  loss: 0.0055214689002396605
# epoch: 30000  loss: 0.0037792143618393583
# softmax, batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), f_score: 0.9622641509433962
# epoch: 6000  loss: 0.07340010183148987
# epoch: 12000  loss: 0.04927722616212974
# epoch: 18000  loss: 0.03232460546378834
# epoch: 24000  loss: 0.02838542803834397
# epoch: 30000  loss: 0.02573550877940547
# regular, batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), f_score: 0.9530516431924884
