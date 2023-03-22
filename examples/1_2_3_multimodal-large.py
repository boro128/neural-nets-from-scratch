from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/multimodal-large-training.csv")
df_test = pd.read_csv("datasets/regression/multimodal-large-test.csv")

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


def execute(batch_size, n_epochs, optimizer, print_every=1000):
    model = mm.Model()

    model.add(mm.LayerDense(1, 10))
    model.add(mm.activation.Sigmoid())
    model.add(mm.LayerDense(10, 5))
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
        n_epochs=n_epochs,
        batch_size=batch_size,
        print_every=print_every,
    )

    out = model.forward(X_test)
    out = scaler_y.reverse_transform(out)
    mse = np.mean(np.square(out - y_test), axis=0).squeeze()
    print(f"{batch_size=}, {n_epochs=}, {optimizer=}, mse: {mse}")

    model.save(
        filename=f"1_2_3_multimodal_large__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


execute(
    batch_size=16,
    n_epochs=25_000,
    optimizer=mm.optim.SGD(lr=0.1, momentum_coeff=0.3),
    print_every=5_000,
)

execute(
    batch_size=1_000,
    n_epochs=100_000,
    optimizer=mm.optim.SGD(lr=0.1, momentum_coeff=0.3),
    print_every=20_000,
)

###################### Experiment notes ######################

########### SGD with momentum ###########

# batch_size=16, n_epochs=25000 mse: 2.8950262440368197, optimizer=SGD, lr=0.1, momentum_coeff=.3 <-----
# batch_size=1000, n_epochs=100000 mse: 5.257448549625033, optimizer=SGD, lr=0.1, momentum_coeff=.3 <-----
# ^^^ lower mse compared to leaning without momentum (1_2_2_multimodal_large.py)

########### RMSProp ###########

# batch_size=1000, n_epochs=100000, optimizer=RMSprop(lr=3e-05, beta=0.9, eps=1e-08), mse: 1261.630773731004

###################### Output ######################

# epoch: 5000  loss: 0.001244963775637699
# epoch: 10000  loss: 0.0012012593767928308
# epoch: 15000  loss: 0.0011904589968652532
# epoch: 20000  loss: 0.0011695026397624592
# epoch: 25000  loss: 0.0011452855590028844
# batch_size=16, n_epochs=25000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 2.8950262440368197
# epoch: 20000  loss: 0.06420094659622107
# epoch: 40000  loss: 0.05751320700424924
# epoch: 60000  loss: 0.0021994731199568027
# epoch: 80000  loss: 0.0019873104931771744
# epoch: 100000  loss: 0.0019148201158069606
# batch_size=1000, n_epochs=100000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 5.257448549625033
