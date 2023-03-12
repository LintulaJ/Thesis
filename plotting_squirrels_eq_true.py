import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

points = 10000
data = np.zeros((points*2, 2))

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

mu_min = 2.2/time_unit
mu_max = 3.2/time_unit

label = ["R" for i in range(points*2)]

for i in range(points):
    data[i,0] = mu_min + i/points * (mu_max-mu_min)
    data[i,1] = K/s * (data[i,0] - lambd) + K
    data[i+points,0] = data[i,0]
    data[i+points,1] = (lambd - data[i,0]) / (data[i,0]) * data[i,1]

    label[i+points] = "I"

df = pd.DataFrame(data, columns = ["mu", "pop"])
df["Population"] = label

sns.set_theme()
plt.axhline(0, color='dimgray')
plt.axvline(0, color='dimgray')
sns.lineplot(x = "mu", y="pop", hue="Population", data=df)
plt.xlim((mu_min, mu_max))
plt.show()