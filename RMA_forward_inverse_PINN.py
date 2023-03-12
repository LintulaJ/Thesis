from deepxde import deepxde as dde
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=20000, maxfun=None, maxls=50)
dde.config.set_random_seed(456)

layer_size = [1] + [32] * 4 + [2]
activation = "swish"
initializer = "Glorot normal"
n_points = 100000

r = 0.3
beta = 0.68
gamma = 1
delta = 0.3
T = 1
K = 3
N_0 = 0.5
P_0 = 0.25

r_inv = dde.Variable(0.5, tf.float64)
beta_inv = dde.Variable(0.5, tf.float64)

def f(x, y):
    N = y[:,0:1]
    P = y[:,1:2]

    dN = dde.grad.jacobian(y, x, i = 0, j = 0)
    dP = dde.grad.jacobian(y, x, i = 1, j = 0)

    return [dN - (r * N * (1 - N / K) - (beta * N * P) / (1 + beta * N * T)),
            dP - (gamma * (beta * N * P) / (1 + beta * N * T) - delta * P)]

def f_inv(x, y):
    N = y[:,0:1]
    P = y[:,1:2]
    dN = dde.grad.jacobian(y, x, i = 0, j = 0)
    dP = dde.grad.jacobian(y, x, i = 1, j = 0)

    return [dN - (r_inv * N * (1 - N / K) - (beta_inv * N * P) / (1 + beta_inv * N * T)),
            dP - (gamma * (beta_inv * N * P) / (1 + beta_inv * N * T) - delta * P)]

def initial_N(x):
    return N_0

def initial_P(x):
    return P_0

def on_initial(x, on_initial):
    return on_initial

time_domain = dde.geometry.TimeDomain(0, 50)

ic_N = dde.icbc.IC(time_domain, initial_N, on_initial, component=0)
ic_P = dde.icbc.IC(time_domain, initial_P, on_initial, component=1)

data = dde.data.TimePDE(
    time_domain,
    f,
    [ic_N, ic_P],
    num_domain=8192,
    num_boundary=128,
    train_distribution="Sobol"
)

net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss="MSE", loss_weights=[1, 1, 1, 1])
model.train(iterations = 8000, display_every = 4000)

model.compile("L-BFGS", loss="MSE", loss_weights=[1, 1, 1, 1])
model.train(display_every = 4000)

points = time_domain.uniform_points(n_points, boundary=True)
pred = model.predict(points)

forward_df = pd.DataFrame(points, columns=["t"])
forward_df["N"] = pred[:,0]
forward_df["P"] = pred[:,1]

sns.lineplot(x = "t", y="N", data=forward_df, label="N")
sns.lineplot(x = "t", y="P", data=forward_df, label="P")
plt.xlabel("t")
plt.ylabel("")
plt.show()


inv_points = points[np.random.choice(np.arange(0, points.shape[0]), 3200)]
inv_points_val = model.predict(inv_points)
inv_points_val += np.random.uniform(low=0.0, high=0.02, size = inv_points_val.shape)

t_domain = dde.geometry.TimeDomain(0,10)
ic_N = dde.icbc.IC(t_domain, initial_N, on_initial, component=0)
ic_P = dde.icbc.IC(t_domain, initial_P, on_initial, component=1)
observe_n = dde.icbc.PointSetBC(inv_points, inv_points_val[:,0:1], component = 0)
observe_p = dde.icbc.PointSetBC(inv_points, inv_points_val[:,1:2], component = 1)
data = dde.data.PDE(
    t_domain,
    f_inv,
    [ic_N, ic_P, observe_n, observe_p],
    num_domain=3200,
    num_boundary=2,
    anchors=inv_points
)

layer_size = [1] + [64] * 5 + [2]
activation = "swish"
initializer = "Glorot normal"

net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
train_variables = [r_inv, beta_inv]
variable = dde.callbacks.VariableValue(
    train_variables, period=1000, filename="variables.dat"
)

model.compile("adam", lr=0.001, external_trainable_variables=train_variables)
model.train(iterations = 10000, callbacks=[variable])

residuals = model.predict(inv_points, f_inv)
residual = np.sqrt(residuals[0]**2 + residuals[1]**2)

loss = np.mean(residual)
max_point_loss = np.max(residual)
print(f"Inverse Loss: {loss}, pw_Loss: {max_point_loss}")