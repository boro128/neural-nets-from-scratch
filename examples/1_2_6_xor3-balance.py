from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/classification/xor3-training.csv")
df_test = pd.read_csv("datasets/classification/xor3-test.csv")

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
    model.add(mm.LayerDense(5, 2))

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
    f_score = mm.functional.fscore(preds, target)
    print(f"{batch_size=}, {n_epochs=}, {optimizer=}, f_score: {f_score}")

    model.save(
        filename=f"1_2_6_xor3-balance__softmax__{repr(optimizer)}__batch_size_{batch_size}__max_patience_{max_patience}.mmodel"
    )


# no regularization
execute(
    32,
    30_000,
    mm.optim.SGD(lr=3e-2, momentum_coeff=0.3, weight_decay=0),
    X_val=None,
    y_val=None,
    print_every=6_000,
)

# L2 regularization
execute(
    32,
    30_000,
    mm.optim.SGD(lr=3e-2, momentum_coeff=0.3, weight_decay=1e-05),
    X_val=None,
    y_val=None,
    print_every=6_000,
)

# early stopping
execute(
    32,
    1_000_000,
    mm.optim.SGD(lr=3e-2, momentum_coeff=0.3),
    X_val=X_val,
    y_val=y_val,
    max_patience=8,
    print_every=6_000,
)

# both
execute(
    32,
    1_000_000,
    mm.optim.SGD(lr=3e-2, momentum_coeff=0.3, weight_decay=1e-05),
    X_val=X_val,
    y_val=y_val,
    max_patience=8,
    print_every=6_000,
)

###################### Experiment notes ######################

########### Baseline ###########
# batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0), f_score: 0.9299065420560747


########### L2 ###########
# batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-05), f_score: 0.9342723004694835
# batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.0001), f_score: 0.9259259259259259
# batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.001), f_score: 0.5835694050991501

########### Early stopping ###########

# Validation loss started to increase -- Early Stopping in epoch 13865
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.0), f_score: 0.9408983451536644

########### Both ###########

# Validation loss started to increase -- Early Stopping in epoch 28235
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-05), f_score: 0.9364705882352942

###################### Output ######################

# epoch: 6000  loss: 0.4073276047437008
# epoch: 12000  loss: 0.04757908265034359
# epoch: 18000  loss: 0.023733259442701143
# epoch: 24000  loss: 0.01607915972157893
# epoch: 30000  loss: 0.011849253476055699
# batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0), f_score: 0.9299065420560747
# epoch: 6000  loss: 0.4463003941714963
# epoch: 12000  loss: 0.050124237836443146
# epoch: 18000  loss: 0.025671688793818443
# epoch: 24000  loss: 0.0182395318932247
# epoch: 30000  loss: 0.014427642664720379
# batch_size=32, n_epochs=30000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-05), f_score: 0.9342723004694835
# epoch: 6000  loss: 0.4073276047437008 val_loss: 0.45931729645075553
# epoch: 12000  loss: 0.04757908265034359 val_loss: 0.06543466750071088
# Validation loss started to increase -- Early Stopping in epoch 13865
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.0), f_score: 0.9408983451536644
# epoch: 6000  loss: 0.4463003941714963 val_loss: 0.484521971864776
# epoch: 12000  loss: 0.050124237836443146 val_loss: 0.06510855794966659
# epoch: 18000  loss: 0.025671688793818443 val_loss: 0.053151920151458296
# epoch: 24000  loss: 0.0182395318932247 val_loss: 0.04842566312778358
# Validation loss started to increase -- Early Stopping in epoch 28235
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-05), f_score: 0.9364705882352942
