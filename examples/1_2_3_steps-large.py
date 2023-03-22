from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/steps-large-training.csv", index_col=0)
df_test = pd.read_csv("datasets/regression/steps-large-test.csv", index_col=0)

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
        n_epochs=n_epochs,
        batch_size=batch_size,
        print_every=print_every,
    )

    out = model.forward(X_test)
    out = scaler_y.reverse_transform(out)
    mse = np.mean(np.square(out - y_test), axis=0).squeeze()
    print(f"{batch_size=}, {n_epochs=}, {optimizer=}, mse: {mse}")

    model.save(
        filename=f"1_2_3_steps_large__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


execute(16, 250_000, mm.optim.SGD(lr=3e-2, momentum_coeff=0.3), 50_000)

###################### Experiment notes ######################

########### SGD with momentum ###########

# execute(32, 100_000, mm.optim.SGD(lr=1e-1, momentum_coeff=.3), 10_000) # 5 mse
# batch_size=16, n_epochs=250000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), mse: 2.707284438438154 <----

########### RMSProp ###########

# batch_size=32, n_epochs=100000, optimizer=RMSprop(lr=0.0001, beta=0.9, eps=1e-08), mse: 20.295642534434922
# batch_size=32, n_epochs=200000, optimizer=RMSprop(lr=3e-05, beta=0.9, eps=1e-08), mse: 28.005904932818723

###################### Output ######################

# epoch: 50000  loss: 0.0009670296630010885
# epoch: 100000  loss: 0.0006923941985121623
# epoch: 150000  loss: 0.0005809891776477899
# epoch: 200000  loss: 0.0004999618606769162
# epoch: 250000  loss: 0.000436838821548369
# batch_size=16, n_epochs=250000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), mse: 2.707284438438154
