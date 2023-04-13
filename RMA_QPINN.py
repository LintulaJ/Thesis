from deepxde import deepxde as dde
import numpy as np
import tensorflow as tf
import cmath
import pandas as pd

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=20000, maxfun=None, maxls=50)
dde.config.set_random_seed(456)

layer_size = [4] + [32] * 4 + [2]
activation = "swish"
initializer = "Glorot normal"
n_points = 500000
eps = 1e-8
tol = 1e-3

r = 0.3
gamma = 1
delta = 0.3
T = 1
K = 3

def f(x, y):
    N = y[:,0:1]
    P = y[:,1:2]
    beta = x[:,0:1]

    dN = dde.grad.jacobian(y, x, i = 0, j = 3)
    dP = dde.grad.jacobian(y, x, i = 1, j = 3)

    return [dN - (r * N * (1 - N / K) - (beta * N * P) / (1 + beta * N * T)),
            dP - (gamma * (beta * N * P) / (1 + beta * N * T) - delta * P)]

def d_dt(x, y):
    dN = dde.grad.jacobian(y, x, i = 0, j = 3)
    dP = dde.grad.jacobian(y, x, i = 1, j = 3)

    return [dN, dP]

def J(x, y):
    J11 = dde.grad.hessian(y, x, i = 3, j = 1, component=0)
    J12 = dde.grad.hessian(y, x, i = 3, j = 2, component=0)
    J21 = dde.grad.hessian(y, x, i = 3, j = 1, component=1)
    J22 = dde.grad.hessian(y, x, i = 3, j = 2, component=1)
    return [J11, J12, J21, J22]

def J_loss(x, y):
    beta = x[:,0:1]
    N = x[:,1:2]
    P = x[:,2:3]

    true_J11 =  r * (1 - 2 * N / K) - (beta * P) / (1 + beta * N * T)**2
    true_J12 = - (beta * N) / (1 + beta * N * T)
    true_J21 = gamma * (beta * P) / (1 + beta * N * T)**2
    true_J22 = gamma * (beta * N) / (1 + beta * N * T) - delta

    J11 = dde.grad.hessian(y, x, i = 3, j = 1, component=0)
    J12 = dde.grad.hessian(y, x, i = 3, j = 2, component=0)
    J21 = dde.grad.hessian(y, x, i = 3, j = 1, component=1)
    J22 = dde.grad.hessian(y, x, i = 3, j = 2, component=1)

    return [J11 - true_J11, J12 - true_J12, J21 - true_J21, J22 - true_J22]

def initial_N(x):
    N_0 = x[:,1:2]
    return N_0

def initial_P(x):
    P_0 = x[:,2:3]
    return P_0

def on_initial(x, on_initial):
    return on_initial

time_domain = dde.geometry.TimeDomain(0, 0.005)
betaxNxP_domain = dde.geometry.geometry_3d.Hypercube((0.3,0,0), (0.8,3,0.8))
space_n_time = dde.geometry.GeometryXTime(betaxNxP_domain, time_domain)

ic_N = dde.icbc.IC(space_n_time, initial_N, on_initial, component=0)
ic_P = dde.icbc.IC(space_n_time, initial_P, on_initial, component=1)

data = dde.data.TimePDE(
    space_n_time,
    f,
    [ic_N, ic_P],
    num_domain=4096,
    num_initial=2048,
    train_distribution="Sobol"
)

net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss="MSE", loss_weights=[1, 1, 5, 5])
model.train(iterations = 8000, display_every = 4000)

model.compile("L-BFGS", loss="MSE", loss_weights=[1, 1, 5, 5])
model.train(display_every = 4000)

points = space_n_time.uniform_points(n_points, boundary=False)
init_points = space_n_time.uniform_initial_points(n_points)

residual_p = model.predict(points, f)
residual = np.sqrt(residual_p[0]**2 + residual_p[1]**2)
dN_residual_p = model.predict(init_points, J_loss)
dN_residual = np.sqrt(dN_residual_p[0]**2 + dN_residual_p[1]**2 + dN_residual_p[2]**2 + dN_residual_p[3]**2)

loss = np.mean(residual)
max_point_loss = np.max(residual)
dN_loss = np.mean(dN_residual)
max_point_dN_loss = np.max(dN_residual)

print(f"Loss: {loss}, pw Loss: {max_point_loss},\nJacobian Loss: {dN_loss}, pw Jacobian Loss: {max_point_dN_loss}")

dt = model.predict(init_points, d_dt)

equilib_mask = np.logical_and(np.isclose(dt[0][:,0], 0,  atol=tol), np.isclose(dt[1][:,0], 0, atol=tol))
equilib_points = init_points[equilib_mask, :]

print(equilib_points)
equilib_df = pd.DataFrame(equilib_points, columns=["beta", "N", "P", "t"])

J11, J12, J21, J22 = model.predict(equilib_points, J)
eig_values = np.zeros((len(equilib_points[:,0]), 2), dtype=np.complex_)
stability =  ["unstable" for i in range((len(equilib_points[:,0])))]

for i in range(len(equilib_points)):
    a = 1
    b = - J11[i] - J22[i]
    c = J11[i] * J22[i] - J12[i] * J21[i]
    eig_values[i, 0] = ((-b + cmath.sqrt(b**2 - 4*a*c)) / (2*a))
    eig_values[i, 1] = ((-b - cmath.sqrt(b**2 - 4*a*c)) / (2*a))

    if (eig_values.real[i, 0] < 0) and (eig_values.real[i, 1] < 0):
        stability[i] = "stable"

equilib_df["equilibria"] = stability
equilib_df.to_csv("_RMA_equilib.csv")

zero_real_i = np.logical_or(np.isclose(eig_values.real[:,0], 0, atol=2e-3), np.isnan(eig_values.real[:,0]))
zero_imag_i = np.logical_or(np.isclose(eig_values.imag[:,0], 0, atol=2e-3), np.isnan(eig_values.imag[:,0]))
purely_imag_i = np.logical_and(zero_real_i, np.invert(zero_imag_i))
print(equilib_points[purely_imag_i])
