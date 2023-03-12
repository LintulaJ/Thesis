from deepxde import deepxde as dde
import numpy as np
import tensorflow as tf
import pandas as pd

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=20000, maxfun=None, maxls=50)
dde.config.set_random_seed(123)

layer_size = [4] + [64] * 3 + [1]
activation = "swish"
initializer = "Glorot normal"
n_points = 30000
eps = 1e-8
tol = 2e-3

def f(x, y):
    N = y[:,0:1]
    r = x[:,1:2]
    c = x[:,2:3]
    dN_dt = dde.grad.jacobian(y, x, i = 0, j = 3)
    return dN_dt - (r * (1 - N) * N + c * N)

def dN_dt(x, y):
    return dde.grad.jacobian(y, x, i = 0, j = 3)

def dN_dt2(x, y):
    return dde.grad.hessian(y, x, i = 3, j = 3)

def df_dN(x, y):
    return dde.grad.hessian(y, x, i = 0, j = 3)

def df_dN_loss(x, y):
    N0 = x[:,0:1]
    r = x[:,1:2]
    c = x[:,2:3]
    return dde.grad.hessian(y, x, i = 3, j = 0) - (r * (1 - 2 * N0) + c)

def true_df_dN(x):
    N0 = x[:,0:1]
    r = x[:1,2]
    c = x[:,2:3]
    return r * (1 - 2 * N0) + c

def degen_cond(x, y):
    N0 = x[:,0:1]
    r = x[:,1:2]
    c = x[:,2:3]
    df_dN0 = dde.grad.hessian(y, x, i = 0, j = 3)
    df_dN0N0 = dde.grad.jacobian(df_dN0, x, j = 0)
    df_dcN0 = dde.grad.jacobian(df_dN0, x, j = 2)
    return [df_dN0N0, df_dcN0]

def initial(x):
    N0 = x[:,0:1]
    return N0

def on_initial(x, on_initial):
    return on_initial

time_domain = dde.geometry.TimeDomain(0, 0.1)
N0xrxc = dde.geometry.geometry_3d.Hypercube((0, 0.1, -0.5), (1, 0.6, 0))

space_n_time = dde.geometry.GeometryXTime(N0xrxc, time_domain)

ic = dde.icbc.IC(space_n_time, initial, on_initial)

data = dde.data.TimePDE(
    space_n_time,
    f,
    [ic],
    num_domain=4096,
    num_initial=1024,
    num_boundary=2,
    train_distribution="Sobol"
)

net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss="MSE", loss_weights=[1, 5])
model.train(iterations = 20000, display_every = 4000)

model.compile("L-BFGS", loss="MSE", loss_weights=[1, 5])
model.train(display_every = 4000)


points = space_n_time.uniform_points(n_points, boundary=False)
init_points = space_n_time.uniform_initial_points(n_points)

residual = np.sqrt(model.predict(points, f)**2)
dN_residual = np.sqrt(model.predict(init_points, df_dN_loss)**2)

loss = np.mean(residual)
max_point_loss = np.max(residual)
dN_loss = np.mean(dN_residual)
max_point_dN_loss = np.max(dN_residual)

print(f"Loss: {loss}, pw Loss: {max_point_loss},\ndN Loss: {dN_loss}, pw dN Loss: {max_point_dN_loss}")

dN = model.predict(init_points, df_dN)
dt = model.predict(init_points, dN_dt)

equilib_mask = np.isclose(dt[:,0], 0, atol=tol)
bifurc_mask = np.logical_and(equilib_mask, np.isclose(dN[:,0], 0, atol=tol))

degen = model.predict(init_points, degen_cond)
non_degen_mask = np.logical_not(np.logical_or(np.isclose(degen[0][:,0], 0, atol=tol), np.isclose(degen[1][:,0], 0, atol=tol)))
transcrit_mask = np.logical_and(bifurc_mask, non_degen_mask)

c_is_025_mask = np.isclose(init_points[:,2], -0.25, atol=0.01)

equilib_data = init_points[np.logical_and(equilib_mask, c_is_025_mask), :]
equilib_df = pd.DataFrame(equilib_data, columns=["N", "r", "c", "t"])
equilib_df.drop(columns=["t"])

unstable_mask = np.where(model.predict(equilib_data, df_dN) < 0, "stable", "unstable")
equilib_df["equilibria"] = unstable_mask
dt_equilib = np.sqrt(dt[np.logical_and(equilib_mask, c_is_025_mask), 0]**2)
equilib_df["res"] = dt_equilib
equilib_df = equilib_df.sort_values(by=["res"])
#equilib_df = equilib_df.drop_duplicates(subset=["r", "label"])
equilib_df = equilib_df.drop(columns=["res"])

transcrit_data = init_points[transcrit_mask, :]
transcrit_df = pd.DataFrame(transcrit_data, columns=["N", "r", "c", "t"])
transcrit_df.drop(columns=["t"])

equilib_df.to_csv("data_equilib_transcrit.csv")
transcrit_df.to_csv("data_bifurc_transcrit.csv")