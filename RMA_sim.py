import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time_step_size = 0.001
time_steps = 1000000

r = 0.2
gamma = 1
delta = 0.2
T = 1
beta = 1

K = 1.5

N_init = 0.3
P_init = 0.15


def dN(N, P):
    return r * N * (1 - N / K) - (beta * N * P) / (1 + beta * N * T)

def dP(N, P):
    return gamma * (beta * N * P) / (1 + beta * N * T) - delta * P

pop = np.zeros(time_steps*2)
label = ["N" for i in range(time_steps*2)]
t = np.zeros(time_steps*2)

pop[0] = N_init
pop[time_steps] = P_init

for i in range(time_steps-1):
    N = pop[i]
    P = pop[i + time_steps]
    pop[i+1] = N + dN(N, P)  * time_step_size
    pop[time_steps+i+1] = P + dP(N, P) * time_step_size


for i in range(time_steps):
    t[i] = time_step_size * i
    t[i + time_steps] = time_step_size * i

    label[time_steps + i] = "P"

df = pd.DataFrame(pop, columns=["Pop"])
df["Species"] = label
df["t"] = t

plt.axhline(0, color='dimgray')
plt.axvline(0, color='dimgray')
sns.lineplot(x = "t", y="Pop", hue="Species", data=df)
plt.xlim((-0.04, 1000))
plt.ylim((-0.04, 0.6))
plt.show()