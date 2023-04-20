from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/classification/rings5-sparse-training.csv")
df_test = pd.read_csv("datasets/classification/rings5-sparse-test.csv")

df_val = df_train.sample(frac=0.15, random_state=123)
df_train = df_train.drop(df_val.index)

X_train = df_train[["x", "y"]].values
y_train = pd.get_dummies(df_train["c"]).values

X_val = df_val[["x", "y"]].values
y_val = pd.get_dummies(df_val["c"]).values

X_test = df_test[["x", "y"]].values
y_test = pd.get_dummies(df_test["c"]).values

scaler_X = mm.preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)


def execute(
    batch_size, n_epochs, optimizer, X_val, y_val, max_patience=None, print_every=1000
):
    model = mm.Model()

    model.add(mm.LayerDense(2, 10))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(10, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 5))

    model.set_optimizer(optimizer)
    model.set_loss(mm.loss.CrossEntropyLoss())

    model.init_weights_xavier()

    model.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        n_epochs=n_epochs,
        batch_size=batch_size,
        max_patience=max_patience,
        print_every=print_every,
    )

    out = model.forward(X_test)
    preds = np.argmax(out, axis=1)
    target = np.argmax(y_test, axis=1)
    f_score = mm.functional.fscore_macro(preds, target)
    print(f"{batch_size=}, {n_epochs=}, {optimizer=}, f_score: {f_score}")

    model.save(
        filename=f"1_2_6_rings5-sparse__{repr(optimizer)}__batch_size_{batch_size}__max_patience_{max_patience}.mmodel"
    )


# no regularization
execute(
    batch_size=32,
    n_epochs=10_000,
    optimizer=mm.optim.SGD(lr=3e-2, momentum_coeff=0.3),
    X_val=None,
    y_val=None,
    max_patience=None,
    print_every=2_000,
)

# L2 regularization
execute(
    batch_size=32,
    n_epochs=10_000,
    optimizer=mm.optim.SGD(lr=3e-2, momentum_coeff=0.3, weight_decay=1e-5),
    X_val=None,
    y_val=None,
    max_patience=None,
    print_every=2_000,
)

# early stopping
execute(
    batch_size=32,
    n_epochs=1_000_000,
    optimizer=mm.optim.SGD(lr=3e-2, momentum_coeff=0.3),
    X_val=X_val,
    y_val=y_val,
    max_patience=7,
    print_every=2_000,
)

# both
execute(
    batch_size=32,
    n_epochs=1_000_000,
    optimizer=mm.optim.SGD(lr=3e-2, momentum_coeff=0.3, weight_decay=1e-5),
    X_val=X_val,
    y_val=y_val,
    max_patience=7,
    print_every=4_000,
)

###################### Experiment notes ######################

########### Baseline ###########

# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.0), f_score: 0.8090923840916409

########### L2 ###########

# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.001), f_score: 0.7715798843483848
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.0001), f_score: 0.8035852538078508
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-05), f_score: 0.8092808272191594
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-06), f_score: 0.8090923840916409
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-07), f_score: 0.8090923840916409

########### Early stopping ###########

# Validation loss started to increase -- Early Stopping in epoch 13170
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.0), f_score: 0.837355375237004

########### Both ###########

# Validation loss started to increase -- Early Stopping in epoch 22395
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-05), f_score: 0.8480877910276281

###################### Output ######################

# epoch: 2000  loss: 1.04856518908274
# epoch: 4000  loss: 0.8043800509196428
# epoch: 6000  loss: 0.5184627555245083
# epoch: 8000  loss: 0.3724647942337265
# epoch: 10000  loss: 0.2751411995788291
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.0), f_score: 0.8090923840916409
# epoch: 2000  loss: 1.049477382419734
# epoch: 4000  loss: 0.805576344064647
# epoch: 6000  loss: 0.5203354011307209
# epoch: 8000  loss: 0.37558667389469574
# epoch: 10000  loss: 0.2774264995133073
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-05), f_score: 0.8092808272191594
# epoch: 2000  loss: 1.04856518908274 val_loss: 0.9376914952435962
# epoch: 4000  loss: 0.8043800509196428 val_loss: 0.7012924635962426
# epoch: 6000  loss: 0.5184627555245083 val_loss: 0.4727353327423721
# epoch: 8000  loss: 0.3724647942337265 val_loss: 0.3692086529484836
# epoch: 10000  loss: 0.2751411995788291 val_loss: 0.3031075094211672
# epoch: 12000  loss: 0.2140281010234993 val_loss: 0.2634811793726602
# Validation loss started to increase -- Early Stopping in epoch 13170
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.0), f_score: 0.837355375237004
# epoch: 4000  loss: 0.805576344064647 val_loss: 0.7020173910213272
# epoch: 8000  loss: 0.37558667389469574 val_loss: 0.3718293478344117
# epoch: 12000  loss: 0.21593701169812932 val_loss: 0.2653497931542852
# epoch: 16000  loss: 0.14106947800313172 val_loss: 0.2209742526795375
# epoch: 20000  loss: 0.09814064761781527 val_loss: 0.21788192760461314
# Validation loss started to increase -- Early Stopping in epoch 22395
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-05), f_score: 0.8480877910276281
