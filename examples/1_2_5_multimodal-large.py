from context import mm

import numpy as np
import pandas as pd
import itertools

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


def execute(
    batch_size, n_epochs, optimizer, n_layers, n_neurons, activation, print_every=1000
):
    model = mm.Model()

    model.add(mm.LayerDense(1, n_neurons))

    for _ in range(n_layers):
        if activation == "sigmoid":
            model.add(mm.activation.Sigmoid())
        elif activation == "relu":
            model.add(mm.activation.ReLu())
        elif activation == "tanh":
            model.add(mm.activation.Tanh())
        model.add(mm.LayerDense(n_neurons, n_neurons))

    if activation == "sigmoid":
        model.add(mm.activation.Sigmoid())
    elif activation == "relu":
        model.add(mm.activation.ReLu())
    elif activation == "tanh":
        model.add(mm.activation.Tanh())

    model.add(mm.LayerDense(n_neurons, 1))

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
    print(
        f"{n_layers=}, {n_neurons=}, {activation=}, {batch_size=}, {n_epochs=}, {optimizer=}, mse: {mse}"
    )

    model.save(
        filename=f"1_2_5_multimodal_large__n_layers_{n_layers}__n_neurons_{n_neurons}__{activation}__{repr(optimizer)}__batch_size_{batch_size}.mmodel"
    )


layers = [1, 2, 3]
neurons = [5, 10]
activations = ["sigmoid", "relu", "tanh"]

for n_layers, n_neurons, activation in itertools.product(layers, neurons, activations):
    execute(
        batch_size=1_000,
        n_epochs=40_000,
        optimizer=mm.optim.SGD(lr=0.1, momentum_coeff=0.3),
        print_every=8_000,
        n_layers=n_layers,
        n_neurons=n_neurons,
        activation=activation,
    )

###################### Output ######################

# epoch: 8000  loss: 0.07711942707911094
# epoch: 16000  loss: 0.07666454351813128
# epoch: 24000  loss: 0.0758979566360464
# epoch: 32000  loss: 0.0749465090689038
# epoch: 40000  loss: 0.0743233391381598
# n_layers=1, n_neurons=5, activation='sigmoid', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 417.26176600181554
# epoch: 8000  loss: 0.2642199474032723
# epoch: 16000  loss: 0.26447896677585886
# epoch: 24000  loss: 0.2656593324688147
# epoch: 32000  loss: 0.2644067794043252
# epoch: 40000  loss: 0.2641578691826233
# n_layers=1, n_neurons=5, activation='relu', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 1393.9720210070914
# epoch: 8000  loss: 0.007225282855558196
# epoch: 16000  loss: 0.0024624322245322167
# epoch: 24000  loss: 0.0023114871843971235
# epoch: 32000  loss: 0.001792321070538204
# epoch: 40000  loss: 0.001932825868149278
# n_layers=1, n_neurons=5, activation='tanh', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 6.845506992096218
# epoch: 8000  loss: 0.013677029202459026
# epoch: 16000  loss: 0.01029090051423943
# epoch: 24000  loss: 0.0027183223786566768
# epoch: 32000  loss: 0.001768905272075952
# epoch: 40000  loss: 0.0016369379441914713
# n_layers=1, n_neurons=10, activation='sigmoid', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 4.325961980056933
# epoch: 8000  loss: 0.017099476186318477
# epoch: 16000  loss: 0.017079188911193158
# epoch: 24000  loss: 0.017003096240846536
# epoch: 32000  loss: 0.016640143690548614
# epoch: 40000  loss: 0.01664961216383767
# n_layers=1, n_neurons=10, activation='relu', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 73.11393498583607
# epoch: 8000  loss: 0.001386705690634145
# epoch: 16000  loss: 0.0012165374969504606
# epoch: 24000  loss: 0.0011994230694962961
# epoch: 32000  loss: 0.0011587972667485523
# epoch: 40000  loss: 0.0011346385561155712
# n_layers=1, n_neurons=10, activation='tanh', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 2.211462715905378
# epoch: 8000  loss: 0.07727518758531222
# epoch: 16000  loss: 0.07430307711961386
# epoch: 24000  loss: 0.07268117452618277
# epoch: 32000  loss: 0.0709245920809224
# epoch: 40000  loss: 0.03577312506955959
# n_layers=2, n_neurons=5, activation='sigmoid', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 189.35616918793434
# epoch: 8000  loss: 0.04271740872244072
# epoch: 16000  loss: 0.040958882324242885
# epoch: 24000  loss: 0.041716237130910085
# epoch: 32000  loss: 0.03916622015623133
# epoch: 40000  loss: 0.038960184961501984
# n_layers=2, n_neurons=5, activation='relu', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 194.40697512122912
# epoch: 8000  loss: 0.0012650684672188969
# epoch: 16000  loss: 0.0011902007238487361
# epoch: 24000  loss: 0.0011792672011859046
# epoch: 32000  loss: 0.0011735828203033585
# epoch: 40000  loss: 0.0011728733722009368
# n_layers=2, n_neurons=5, activation='tanh', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 1.8182781810851176
# epoch: 8000  loss: 0.012767987541525343
# epoch: 16000  loss: 0.0026692645458918727
# epoch: 24000  loss: 0.0016634340247942184
# epoch: 32000  loss: 0.0016191512987842802
# epoch: 40000  loss: 0.0015572294655769687
# n_layers=2, n_neurons=10, activation='sigmoid', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 3.881363313630863
# epoch: 8000  loss: 0.0017126102025716134
# epoch: 16000  loss: 0.001628421277547572
# epoch: 24000  loss: 0.0015393944830304521
# epoch: 32000  loss: 0.001501868249757299
# epoch: 40000  loss: 0.0015639149388877894
# n_layers=2, n_neurons=10, activation='relu', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 5.027455521481073
# epoch: 8000  loss: 0.001183547444333926
# epoch: 16000  loss: 0.001162055573915751
# epoch: 24000  loss: 0.0011359621532266884
# epoch: 32000  loss: 0.0010831722809767701
# epoch: 40000  loss: 0.0008906206336202177
# n_layers=2, n_neurons=10, activation='tanh', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 1.8975339652888992
# epoch: 8000  loss: 0.07623780163137113
# epoch: 16000  loss: 0.06417925533700705
# epoch: 24000  loss: 0.05967202758451031
# epoch: 32000  loss: 0.0018994534582676664
# epoch: 40000  loss: 0.0014466515135891443
# n_layers=3, n_neurons=5, activation='sigmoid', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 2.5937953462824663
# epoch: 8000  loss: 0.018675030227452562
# epoch: 16000  loss: 0.01727478888339335
# epoch: 24000  loss: 0.017664000232124267
# epoch: 32000  loss: 0.016310945554378502
# epoch: 40000  loss: 0.016218033670738867
# n_layers=3, n_neurons=5, activation='relu', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 73.4550140576349
# epoch: 8000  loss: 0.0018158700261649767
# epoch: 16000  loss: 0.0015186035608735677
# epoch: 24000  loss: 0.0014963523214366285
# epoch: 32000  loss: 0.0014282295263417
# epoch: 40000  loss: 0.0013742374915358493
# n_layers=3, n_neurons=5, activation='tanh', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 2.334191143285815
# epoch: 8000  loss: 0.0036555961457705117
# epoch: 16000  loss: 0.0014059462887955887
# epoch: 24000  loss: 0.0012916055699196244
# epoch: 32000  loss: 0.0012101890847278029
# epoch: 40000  loss: 0.0011799092224876384
# n_layers=3, n_neurons=10, activation='sigmoid', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 1.770656598203751
# epoch: 8000  loss: 0.00197125897717069
# epoch: 16000  loss: 0.001352671532342175
# epoch: 24000  loss: 0.001289943874560418
# epoch: 32000  loss: 0.0012693436275816065
# epoch: 40000  loss: 0.0012645011329197216
# n_layers=3, n_neurons=10, activation='relu', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 2.5022437517738254
# epoch: 8000  loss: 0.0011583224431024023
# epoch: 16000  loss: 0.0009894313617156852
# epoch: 24000  loss: 0.0006888564978501718
# epoch: 32000  loss: 0.0005063125393067361
# epoch: 40000  loss: 0.0004875522981110216
# n_layers=3, n_neurons=10, activation='tanh', batch_size=1000, n_epochs=40000, optimizer=SGD(lr=0.1, momentum_coeff=0.3), mse: 0.7438690115452115


# best results:
# 3 layers, 10 neurons, tanh
# 3 layers, 10 neurons, sigmoid
