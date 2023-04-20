from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/classification/rings3-balance-training.csv")
df_test = pd.read_csv("datasets/classification/rings3-balance-test.csv")

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
        filename=f"1_2_6_rings3-balance__{repr(optimizer)}__batch_size_{batch_size}__max_patience_{max_patience}.mmodel"
    )


# no regularization
execute(
    batch_size=32,
    n_epochs=10_000,
    optimizer=mm.optim.SGD(lr=3e-2, momentum_coeff=0.3, weight_decay=0),
    X_val=None,
    y_val=None,
    max_patience=None,
    print_every=2_000,
)

# L2 regularization
execute(
    batch_size=32,
    n_epochs=10_000,
    optimizer=mm.optim.SGD(lr=3e-2, momentum_coeff=0.3, weight_decay=1e-6),
    X_val=None,
    y_val=None,
    max_patience=None,
    print_every=2_000,
)

# early stopping
execute(
    batch_size=32,
    n_epochs=1_000_000,
    optimizer=mm.optim.SGD(lr=3e-2, momentum_coeff=0.3, weight_decay=0),
    X_val=X_val,
    y_val=y_val,
    max_patience=6,
    print_every=2_000,
)

# both
execute(
    batch_size=32,
    n_epochs=1_000_000,
    optimizer=mm.optim.SGD(lr=3e-2, momentum_coeff=0.3, weight_decay=1e-6),
    X_val=X_val,
    y_val=y_val,
    max_patience=6,
    print_every=2_000,
)

###################### Experiment notes ######################

########### Baseline ###########
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0), f_score: 0.8121685677074734

########### L2 ###########

# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0.0001), f_score: 0.729976325739949
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-05), f_score: 0.8043142080012705
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-06), f_score: 0.8109811668792973
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-07), f_score: 0.8115043281252828
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=5e-05), f_score: 0.7622994881929227
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=5e-06), f_score: 0.8070483915343377
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=5e-07), f_score: 0.8108400862412043

########### Early stopping ###########

# Validation loss started to increase -- Early Stopping in epoch 8089
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0), f_score: 0.8053483499284599

########### Both ###########

# Validation loss started to increase -- Early Stopping in epoch 8089
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-06), f_score: 0.80196689103047

###################### Output ######################

# epoch: 2000  loss: 0.10634983198579587
# epoch: 4000  loss: 0.06334033711116714
# epoch: 6000  loss: 0.05209230607807777
# epoch: 8000  loss: 0.045592575275530726
# epoch: 10000  loss: 0.041502232059887456
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0), f_score: 0.8121685677074734
# epoch: 2000  loss: 0.10660722812469856
# epoch: 4000  loss: 0.063619027197468674
# epoch: 6000  loss: 0.05236553382325521
# epoch: 8000  loss: 0.045893313299844724
# epoch: 10000  loss: 0.04169901698510362
# batch_size=32, n_epochs=10000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-06), f_score: 0.8109811668792973
# epoch: 2000  loss: 0.10634983198579587 val_loss: 0.1331364752721296
# epoch: 4000  loss: 0.06334033711116714 val_loss: 0.12419507625326157
# epoch: 6000  loss: 0.05209230607807777 val_loss: 0.12718154193637807
# epoch: 8000  loss: 0.045592575275530726 val_loss: 0.12543054837302078
# Validation loss started to increase -- Early Stopping in epoch 8089
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=0), f_score: 0.8053483499284599
# epoch: 2000  loss: 0.10660722812469856 val_loss: 0.13334261641879938
# epoch: 4000  loss: 0.06361902719746867 val_loss: 0.12415943187128685
# epoch: 6000  loss: 0.05236553382325521 val_loss: 0.1268848817027058
# epoch: 8000  loss: 0.045893313299844724 val_loss: 0.12474621810105281
# Validation loss started to increase -- Early Stopping in epoch 8089
# batch_size=32, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3, weight_decay=1e-06), f_score: 0.80196689103047
