from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/multimodal-sparse-training.csv")
df_test = pd.read_csv("datasets/regression/multimodal-sparse-test.csv")

df_val = df_train.sample(frac=0.15, random_state=123)
df_train = df_train.drop(df_val.index)

X_train = df_train["x"].values.reshape((-1, 1))
y_train = df_train["y"].values.reshape((-1, 1))

X_val = df_val["x"].values.reshape((-1, 1))
y_val = df_val["y"].values.reshape((-1, 1))

X_test = df_test["x"].values.reshape((-1, 1))
y_test = df_test["y"].values.reshape((-1, 1))

scaler_X = mm.preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

scaler_y = mm.preprocessing.StandardScaler()
y_train_transf = scaler_y.fit_transform(y_train)
y_val_transf = scaler_y.transform(y_val)
y_test_transf = scaler_y.transform(y_test)


def execute(
    batch_size, n_epochs, optimizer, X_val, y_val, max_patience=None, print_every=1000
):
    model = mm.Model()

    model.add(mm.LayerDense(1, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 5))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(5, 1))

    model.set_optimizer(optimizer)
    model.set_loss(mm.loss.MSELoss())

    model.init_weights_xavier()

    model.train(
        X_train,
        y_train_transf,
        X_val=X_val,
        y_val=y_val,
        n_epochs=n_epochs,
        batch_size=batch_size,
        max_patience=max_patience,
        print_every=print_every,
    )

    out = model.forward(X_test)
    out = scaler_y.reverse_transform(out)
    mse = np.mean(np.square(out - y_test), axis=0).squeeze()
    print(f"{batch_size=}, {n_epochs=}, {optimizer=}, mse: {mse}")

    model.save(
        filename=f"1_2_6_multimodal_sparse__{repr(optimizer)}__batch_size_{batch_size}__max_patience_{max_patience}.mmodel"
    )


# no regularization
execute(
    batch_size=1,
    n_epochs=10_000,
    optimizer=mm.optim.SGD(lr=0.1),
    X_val=None,
    y_val=None,
    max_patience=None,
    print_every=2_000,
)

# weight decay
execute(
    batch_size=1,
    n_epochs=10_000,
    optimizer=mm.optim.SGD(lr=0.1, weight_decay=1e-4),
    X_val=None,
    y_val=None,
    max_patience=None,
    print_every=2_000,
)

# early stopping
execute(
    batch_size=1,
    n_epochs=1_000_000,
    optimizer=mm.optim.SGD(lr=0.1),
    X_val=X_val,
    y_val=y_val_transf,
    max_patience=7,
    print_every=8_000,
)

###################### Experiment notes ######################

########### Baseline ###########
# batch_size=1, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.0), mse: 667.1647614624802

########### L2 regularization ###########
# batch_size=1, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.1), mse: 6915.401484521921
# batch_size=1, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.01), mse: 4920.0508371576325
# batch_size=1, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.001), mse: 655.1129182597253
# batch_size=1, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.0001), mse: 581.1389358219869
# batch_size=1, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=1e-05), mse: 614.9082404783685
# batch_size=1, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=1e-06), mse: 663.2781137429739


# batch_size=None, n_epochs=100000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.0), mse: 493.84391769104207
# batch_size=None, n_epochs=100000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=1e-07), mse: 493.9412058553864
# batch_size=None, n_epochs=100000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=1e-06), mse: 494.8260690540951

# batch_size=None, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0), mse: 721.133614608321
# batch_size=None, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=1e-05), mse: 723.5137795004473
# batch_size=None, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.0001), mse: 744.3759686405415
# batch_size=None, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=1e-06), mse: 721.3722811920877

########### Early stopping ###########

# Validation loss started to increase -- Early Stopping in epoch 41503
# batch_size=1, n_epochs=1000000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.0), mse: 344.63004368504585

###################### Output ######################

# epoch: 2000  loss: 0.21351084703597256
# epoch: 4000  loss: 0.08115575373127305
# epoch: 6000  loss: 0.0740849429256099
# epoch: 8000  loss: 0.06633322091629387
# epoch: 10000  loss: 0.07373963346444247
# batch_size=1, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.0), mse: 667.1647614624802
# epoch: 2000  loss: 0.08220412082504164
# epoch: 4000  loss: 0.08689409174658395
# epoch: 6000  loss: 0.07791587544534295
# epoch: 8000  loss: 0.0790085808675191
# epoch: 10000  loss: 0.08723703258974755
# batch_size=1, n_epochs=10000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.0001), mse: 581.1389358219869
# epoch: 8000  loss: 0.06633322091629387 val_loss: 0.013054034680676558
# epoch: 16000  loss: 0.006591267246849109 val_loss: 0.0014206350217504035
# epoch: 24000  loss: 0.006020571128964958 val_loss: 0.006319166307540885
# epoch: 32000  loss: 0.005034088346080156 val_loss: 0.00332140009234543
# epoch: 40000  loss: 0.004819146844418267 val_loss: 0.0037307917735681637
# Validation loss started to increase -- Early Stopping in epoch 41503
# batch_size=1, n_epochs=1000000, optimizer=SGD(lr=0.1, momentum_coeff=0.0, weight_decay=0.0), mse: 344.63004368504585
