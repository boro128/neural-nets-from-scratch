from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/classification/rings3-regular-training.csv")
df_test = pd.read_csv("datasets/classification/rings3-regular-test.csv")

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
    model.add(mm.LayerDense(5, 3))

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
    print(f"softmax, {batch_size=}, {n_epochs=}, {optimizer=}, f_score: {f_score}")

    model.save(
        filename=f"1_2_4_rings3-regular__softmax__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


execute_softmax(32, 10_000, mm.optim.SGD(lr=3e-3, momentum_coeff=0.3), 2_000)


###################### Output ######################

# epoch: 2000  loss: 0.9670244200971337
# epoch: 4000  loss: 0.906943761193602
# epoch: 6000  loss: 0.4972971900326316
# epoch: 8000  loss: 0.32925420805519395
# epoch: 10000  loss: 0.26268313481367395
# softmax, batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.003, momentum_coeff=0.3), f_score: 0.8962524408147668

