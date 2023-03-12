from deepxde import deepxde as dde
import numpy as np
import tensorflow as tf
import cmath
import pandas as pd

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=20000, maxfun=None, maxls=50)
dde.config.set_random_seed(789)

layer_size = [5] + [64] * 3 + [3]
activation = "swish"
initializer = "He normal"
eps = 1e-4
eq_tol = 2e-4
bifurc_tol = 1e-3

time_unit = 3

r = 1.2 / time_unit
E = 1.25 
k = 0.61 / time_unit
c = 1 
s = 1 / time_unit
K = 0.5 
a = 1.65 / time_unit
beta = 3.27 / time_unit
lambd = 3.27 / time_unit

def f(x, y):
    mu = x[:,0:1]
    G = y[:,0:1]
    R = y[:,1:2]
    I = y[:,2:3]

    dGdt = dde.grad.jacobian(y, x, i = 0, j = 4)
    dRdt = dde.grad.jacobian(y, x, i = 1, j = 4)
    dIdt = dde.grad.jacobian(y, x, i = 2, j = 4)

    return [dGdt - (r * G * (1 - G / E) - (k * R * G) / (G + c * (R + I))),
            dRdt - (s * R * (1 - R / K) - (a * R * G) / (G + c * (R + I)) - R * (lambd * I + beta * G) / (R + I + G)),
            dIdt - (R * (lambd * I + beta * G) / (R + I + G) - mu * I)]

def d_dt(x, y):
    dG = dde.grad.jacobian(y, x, i = 0, j = 4)
    dR = dde.grad.jacobian(y, x, i = 1, j = 4)
    dI = dde.grad.jacobian(y, x, i = 2, j = 4)

    return [dG, dR, dI]

def J(x, y):
    J11 = dde.grad.hessian(y, x, i = 4, j = 1, component=0)
    J12 = dde.grad.hessian(y, x, i = 4, j = 2, component=0)
    J13 = dde.grad.hessian(y, x, i = 4, j = 3, component=0)
    J21 = dde.grad.hessian(y, x, i = 4, j = 1, component=1)
    J22 = dde.grad.hessian(y, x, i = 4, j = 2, component=1)
    J23 = dde.grad.hessian(y, x, i = 4, j = 3, component=1)
    J31 = dde.grad.hessian(y, x, i = 4, j = 1, component=2)
    J32 = dde.grad.hessian(y, x, i = 4, j = 2, component=2)
    J33 = dde.grad.hessian(y, x, i = 4, j = 3, component=2)
    return [J11, J12, J13, J21, J22, J23, J31, J32, J33]

def J_loss(x, y):
    mu = x[:,0:1]
    G = x[:,1:2]
    R = x[:,2:3]
    I = x[:,3:4]

    J11 = dde.grad.hessian(y, x, i = 4, j = 1, component=0)
    J12 = dde.grad.hessian(y, x, i = 4, j = 2, component=0)
    J13 = dde.grad.hessian(y, x, i = 4, j = 3, component=0)
    J21 = dde.grad.hessian(y, x, i = 4, j = 1, component=1)
    J22 = dde.grad.hessian(y, x, i = 4, j = 2, component=1)
    J23 = dde.grad.hessian(y, x, i = 4, j = 3, component=1)
    J31 = dde.grad.hessian(y, x, i = 4, j = 1, component=2)
    J32 = dde.grad.hessian(y, x, i = 4, j = 2, component=2)
    J33 = dde.grad.hessian(y, x, i = 4, j = 3, component=2)

    t_J11 = r * (1 - 2 * G / E) - (k * R) / (G + c * (R + I)) + (k * R * G) / (G + c * (R + I))**2
    t_J12 = - (k * G) / (G + c * (R + I)) + (c * k * R * G) / (G + c * (R + I))**2
    t_J13 = (c * k * R * G) / (G + c * (R + I))**2
    t_J21 = - (a * R) / (G + c * (R + I)) + (a * R * G)  / (G + c * (R + I))**2 - R * (beta) / (R + I + G) + R * (lambd * I + beta * G) / (R + I + G)**2 
    t_J22 = s * (1 - 2 * R / K) - (a * G) / (G + c * (R + I)) + (c * a * R * G) / (G + c * (R + I))**2 - (beta * G + lambd * I) / (R + I + G) + R * (lambd * I + beta * G) / (R + I + G)**2 
    t_J23 = c * (a * R * G) / (G + c * (R + I))**2 - R * (lambd) / (R + I + G) + R * (lambd * I + beta * G) / (R + I + G)**2
    t_J31 = R * (beta) / (R + I + G) - R * (lambd * I + beta * G) / (R + I + G)**2 
    t_J32 = (beta * G + lambd * I) / (R + I + G) - R * (lambd * I + beta * G) / (R + I + G)**2 
    t_J33 = R * (lambd) / (R + I + G) - R * (lambd * I + beta * G) / (R + I + G)**2 - mu

    return [J11 - t_J11, J12 - t_J12, J13 - t_J13, 
            J21 - t_J21, J22 - t_J22, J23 - t_J23,
            J31 - t_J31, J32 - t_J32, J33 - t_J33]

def initial_G(x):
    G_0 = x[:,1:2]
    return G_0

def initial_R(x):
    R_0 = x[:,2:3]
    return R_0

def initial_I(x):
    I_0 = x[:,3:4]
    return I_0

def on_initial(x, on_initial):
    return on_initial


time_domain = dde.geometry.TimeDomain(0, 0.005)
muxGxRxI = dde.geometry.geometry_nd.Hypercube((2.2/time_unit, 0, 0.01, 0), (3.2/time_unit, 0.01, K, 0.05))
space_n_time = dde.geometry.GeometryXTime(muxGxRxI, time_domain)

ic_G = dde.icbc.IC(space_n_time, initial_G, on_initial, component=0)
ic_R = dde.icbc.IC(space_n_time, initial_R, on_initial, component=1)
ic_I = dde.icbc.IC(space_n_time, initial_I, on_initial, component=2)

data = dde.data.TimePDE(
    space_n_time,
    f,
    [ic_G, ic_R, ic_I],
    num_domain=8**5,
    num_initial=7**4,
    train_distribution="Sobol"
)

net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

model.compile("adam", lr=2e-3, loss="MSE", loss_weights=[1, 1, 1, 5, 5, 5])
model.train(iterations = 8000, display_every = 2000)

model.compile("L-BFGS", loss="MSE", loss_weights=[1, 1, 1, 2, 2, 2])
model.train(display_every = 2000)

init_points = space_n_time.random_initial_points(25**4)

residual_p = model.predict(init_points, f)
residual = np.sqrt(residual_p[0]**2 + residual_p[1]**2 + residual_p[2]**2)
dN_residual_p = model.predict(init_points, J_loss)
dN_residual = np.zeros_like(dN_residual_p[0])
for i in range(0, 9):
    dN_residual += dN_residual_p[i]**2

dN_residual = np.sqrt(dN_residual)

loss = np.mean(residual)
max_point_loss = np.max(residual)
dN_loss = np.mean(dN_residual)
max_point_dN_loss = np.max(dN_residual)
max = np.argmax(dN_residual)

print(init_points[max, :])

print(f"Loss: {loss}, pw Loss: {max_point_loss},\nJacobian Loss: {dN_loss}, pw Jacobian Loss: {max_point_dN_loss}")

dt = model.predict(init_points, d_dt)

equilib_mask = np.logical_and(np.logical_and(np.isclose(dt[0][:,0], 0,  atol=eq_tol), np.isclose(dt[1][:,0], 0, atol=eq_tol)), np.isclose(dt[2][:,0], 0,  atol=eq_tol))
equilib_points = init_points[equilib_mask, :]
print(equilib_points)

equilib_df = pd.DataFrame(equilib_points, columns=["mu", "G", "R", "I", "t"])
equilib_df.drop(columns=["t"])
equilib_df.to_csv("equilib_data_squirrels.csv")

jacobian_pred = model.predict(equilib_points, J)
jacobian = np.zeros((3, 3))
eigvals = np.linalg.eigvals(jacobian)
zero_eigvals = 0

for i in range(len(equilib_points)):
    for j in range(3):
        for k in range(3):
            jacobian[j, k] = jacobian_pred[3*j+k][i, :]
    
    eigvals = np.linalg.eigvals(jacobian)
    zero_eigvals = 0
    for j in range(3):
        if (np.real(eigvals[j])**2 < bifurc_tol):
            print(eigvals[j])
            zero_eigvals = zero_eigvals + 1

    if (zero_eigvals == 2):
        print(f"Hopf bifurcation at: {equilib_points[i, :]}")

