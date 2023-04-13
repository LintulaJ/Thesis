import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


points = 1000
data = np.zeros((points, 2))
label = ["g(x)" for i in range(points)]

r = 0.3
gamma = 1
delta = 0.3
T = 1
K = 3

for i in range(points):
    beta = 0.5 * i/points + 0.3
    data[i,0] = delta / (beta * (gamma - delta*T))
    data[i,1] = r * (1 - data[i,0]/K) * ((1 + beta * data[i,0] * T) / beta)
    if data[i,0] > ((K - 1/(beta*T))/2):
        label[i] = "stable"
    else:
        label[i] = "unstable"

df = pd.DataFrame(data, columns = ["N", "P"])
df["equilibria"] = label

sns.set_theme()
sns.lineplot(x = "N", y="P", data=df, hue="equilibria")
plt.show()