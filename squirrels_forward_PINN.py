from deepxde import deepxde as dde
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=8000, maxfun=None, maxls=50)
dde.config.set_random_seed(78910)

layer_size = [2] + [64] * 3 + [3]
activation = "swish"
initializer = "Glorot normal"
n_points = 100**2
eps = 1e-8
tol = 5e-3

r = 1
E = 1.25
k = 0.61
s = 1
c = 1
K = 0.5
a = 1.65
beta = 3.27
lambd = 3.27

G_0 = 0.1
R_0 = K
I_0 = 0

def f(x, y):
    mu = x[:,0:1]
    G = y[:,0:1]
    R = y[:,1:2]
    I = y[:,2:3]

    dGdt = dde.grad.jacobian(y, x, i = 0, j = 1)
    dRdt = dde.grad.jacobian(y, x, i = 1, j = 1)
    dIdt = dde.grad.jacobian(y, x, i = 2, j = 1)

    return [dGdt - (r * G * (1 - G / E) - (k * R * G) / (G + c * (R + I))),
            dRdt - (s * R * (1 - R / K) - (a * R * G) / (G + c * (R + I)) - R * (lambd * I + beta * G) / (R + I + G)),
            dIdt - (R * (lambd * I + beta * G) / (R + I + G) - mu * I)]

def initial_G(x):
    return G_0

def initial_R(x):
    return R_0

def initial_I(x):
    return I_0

def on_initial(x, on_initial):
    return on_initial

time_domain = dde.geometry.TimeDomain(0, 2)
mu_domain = dde.geometry.geometry_1d.Interval(1, 10)
space_n_time = dde.geometry.GeometryXTime(mu_domain, time_domain)

ic_G = dde.icbc.IC(space_n_time, initial_G, on_initial, component=0)
ic_R = dde.icbc.IC(space_n_time, initial_R, on_initial, component=1)
ic_I = dde.icbc.IC(space_n_time, initial_I, on_initial, component=2)

data = dde.data.TimePDE(
    space_n_time,
    f,
    [ic_G, ic_R, ic_I],
    num_domain=64**2,
    num_initial=1024,
    train_distribution="Sobol"
)

net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

#model.compile("adam", lr=1e-3, loss="MSE", loss_weights=[1, 1, 1, 20, 20, 20])
#model.train(iterations = 4000, display_every = 1000)

model.compile("L-BFGS", loss="MSE", loss_weights=[1, 1, 1, 5, 20, 5])
model.train(display_every = 1000)


points = space_n_time.uniform_points(n_points, boundary=False)
pred = model.predict(points)

df = pd.DataFrame(points, columns=["mu", "t"])
df["G"] = pred[:,0]
df["R"] = pred[:,1]
df["I"] = pred[:,2]

df.to_csv("_sqrl_sim_3.csv")