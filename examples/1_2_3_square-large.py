from context import mm

import numpy as np
import pandas as pd

df_train = pd.read_csv("datasets/regression/square-large-training.csv", index_col=0)
df_test = pd.read_csv("datasets/regression/square-large-test.csv", index_col=0)

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
    model.add(mm.LayerDense(10, 1))

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
        filename=f"1_2_3_square_large__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


execute(16, 1_500_000, mm.optim.SGD(lr=0.03, momentum_coeff=0.3), 300_000)

execute(16, 1_500_000, mm.optim.RMSprop(lr=8e-6, beta=0.9), 300_000)

###################### Experiment notes ######################

########### SGD with momentum ###########

## 2 hidden layers 5 neurons each

# batch_size=16, n_epochs=50000 mse: 15.708729871414738, lr=0.1, momentum_coeff=.3
# batch_size=16, n_epochs=100000 mse: 14.725674206408767, lr=0.05, momentum_coeff=.3

## 1 hidden layer 5 neurons
# batch_size=16, n_epochs=100000, optimizer=SGD(lr=0.05, momentum_coeff=0.3), mse: 7.654519635875013
# batch_size=16, n_epochs=500000, optimizer=SGD(lr=0.05, momentum_coeff=0.3), mse: 2.452663999248446
# batch_size=16, n_epochs=1000000, optimizer=SGD(lr=0.05, momentum_coeff=0.3), mse: 2.4502404636153816
# batch_size=16, n_epochs=1000000, optimizer=SGD(lr=0.01, momentum_coeff=0.3), mse: 4.3044858918985875

## 1 hidden layer 10 neurons
# batch_size=16, n_epochs=1000000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), mse: 1.22982585271382
# batch_size=16, n_epochs=1500000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), mse: 0.9903636753641945 <-------

########### RMSProp ###########

## 2 hidden layers 5 neurons each

# batch_size=16, n_epochs=50000 mse: 36.0172193401734, lr=0.001, beta=0.9
# batch_size=16, n_epochs=50000 mse: 17.555457261509044, lr=0.0001, beta=0.9
# batch_size=16, n_epochs=100000 mse: 29.02696764620551, lr=1e-5, beta=0.9
# ^^^ loss was constantly dropping, trying more epochs is a good idea
# batch_size=16, n_epochs=150000 mse: 15.241638129671246, lr=3e-5, beta=0.9
# batch_size=16, n_epochs=100000 mse: 19.051338594526804, lr=3e-5, beta=0.9
# batch_size=16, n_epochs=200000 mse: 22.462643618783833, lr=1e-5, beta=0.9
# batch_size=16, n_epochs=200000 mse: 42.37192981837577, lr=3e-6, beta=0.9

## 1 hidden layer 10 neurons

# batch_size=16, n_epochs=1500000, optimizer=RMSprop(lr=8e-06, beta=0.9, eps=1e-08), mse: 0.46165096409156803 <-----

###################### Output ######################

# epoch: 300000  loss: 7.183449753779845e-07
# epoch: 600000  loss: 1.5557316479632196e-07
# epoch: 900000  loss: 5.618705967938853e-08
# epoch: 1200000  loss: 2.9130152373757813e-08
# epoch: 1500000  loss: 2.041671566216251e-08
# batch_size=16, n_epochs=1500000, optimizer=SGD(lr=0.03, momentum_coeff=0.3), mse: 0.9903636753641945
# epoch: 300000  loss: 2.404182696599145e-07
# epoch: 600000  loss: 2.5465942099685765e-08
# epoch: 900000  loss: 1.4646147764493165e-08
# epoch: 1200000  loss: 9.748719273877257e-09
# epoch: 1500000  loss: 7.393498191206115e-09
# batch_size=16, n_epochs=1500000, optimizer=RMSprop(lr=8e-06, beta=0.9, eps=1e-08), mse: 0.46165096409156803
