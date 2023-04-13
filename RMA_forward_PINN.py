from deepxde import deepxde as dde
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import multiprocessing

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=8000, maxfun=None, maxls=50)
dde.config.set_random_seed(99)

layer_size = [1] + [32] * 4 + [2]
activation = "sin"
initializer = "Glorot normal"
n_points = 1000
eps = 1e-8
tol = 2e-3

r = 0.1
beta = 1
gamma = 1
delta = 0.2
T = 1
K = 1.5 # Change this 
N_0 = 0.3
P_0 = 0.15

target_file = "_RMA_forward_K15.csv"
iteration = 0

def f(x, y):
    N = y[:,0:1]
    P = y[:,1:2]

    dN = dde.grad.jacobian(y, x, i = 0, j = 0)
    dP = dde.grad.jacobian(y, x, i = 1, j = 0)

    return [dN - (r * N * (1 - N / K) - (beta * N * P) / (1 + beta * N * T)),
            dP - (gamma * (beta * N * P) / (1 + beta * N * T) - delta * P)]


def on_initial(x, on_initial):
    return on_initial


def forward(q):
    N_0 = q.get()
    P_0 = q.get()
    it = q.get()
    time_domain = dde.geometry.TimeDomain(0, 25)

    ic_N = dde.icbc.IC(time_domain, lambda x: N_0, on_initial, component=0)
    ic_P = dde.icbc.IC(time_domain, lambda x: P_0, on_initial, component=1)

    data = dde.data.TimePDE(
        time_domain,
        f,
        [ic_N, ic_P],
        num_domain=8192,
        num_boundary=32,
        train_distribution="Sobol"
    )

    net = dde.nn.FNN(layer_size, activation, initializer)
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss="MSE", loss_weights=[1, 1, 1, 1])
    model.train(iterations = 1000, display_every = 1000)

    model.compile("L-BFGS", loss="MSE", loss_weights=[1, 1, 1, 1])
    model.train(display_every = 8000)

    points = time_domain.uniform_points(n_points, boundary=True)

    res = model.predict(points, f)
    mean_res = np.mean(np.sqrt(res[0]**2 + res[1]**2))
    print(f"mean res: {mean_res}")
    
    if (mean_res < 1e-4):
        pred = model.predict(points)
        df = pd.DataFrame(points, columns=["t"])
        df["N"] = pred[:,0]
        df["P"] = pred[:,1]
        df["t"] += it * 25
        df.to_csv(target_file, mode="a", header=False, index=False)
        q.put(it + 1, False)
        q.put(pred[-1, 0])
        q.put(pred[-1, 1])
    else:
        q.put(it, False)
        q.put(N_0, False)
        q.put(P_0, False)  


if __name__ == '__main__':
    
    header = ["t", "N", "P"]
    file = open(target_file, "w")
    writer = csv.writer(file)
    writer.writerow(header)
    file.close()

    q = multiprocessing.Queue(3)
    q.put(N_0)
    q.put(P_0)

    while (iteration < 40):
        print(f"iteration {iteration}")
        q.put(iteration, False)
        p = multiprocessing.Process(target=forward(q))
        p.start()
        p.join(1000)
        iteration = q.get()
        p.terminate()
        p.close()

    df = pd.read_csv(target_file)
    sns.lineplot(x = "t", y="N", data=df, label="N")
    sns.lineplot(x = "t", y="P", data=df, label="P")
    plt.xlabel("t")
    plt.ylabel("")
    plt.show()