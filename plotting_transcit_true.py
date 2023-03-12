
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

points = 10000
data = np.zeros((points*2, 2))
label = ["g(x)" for i in range(points*2)]

for i in range(points):
    data[i,0] = 2*i/points - 1
    data[i,1] = 2*i/points - 1
    data[i+points,0] = 2*i/points - 1
    data[i+points,1] = 0
    if i < 5000:
        label[i+points] = "stable"
        label[i] = "unstable"
    else:
        label[i+points] = "unstable"
        label[i] = "stable"

df = pd.DataFrame(data, columns = ["r-c", "x"])
df["equilibria"] = label

sns.set_theme()
plt.axhline(0, color='dimgray')
plt.axvline(0, color='dimgray')
sns.lineplot(x = "r-c", y="x", hue="equilibria", data=df)
plt.xlim((-1, 1))
plt.show()