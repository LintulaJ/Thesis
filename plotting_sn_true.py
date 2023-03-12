
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

points = 10000
data = np.zeros((points*2, 2))
label = ["g(x)" for i in range(points*2)]

for i in range(points):
    data[i,0] = i/points
    data[i,1] = np.sqrt(i/points)
    data[i+points,0] = i/points
    data[i+points,1] = -np.sqrt(i/points)

    label[i+points] = "unstable"
    label[i] = "stable"

df = pd.DataFrame(data, columns = ["alpha", "x"])
df["equilibria"] = label

sns.set_theme()
plt.axhline(0, color='dimgray')
plt.axvline(0, color='dimgray')
sns.lineplot(x = "alpha", y="x", hue="equilibria", data=df)
plt.xlim((-0.2, 1))
plt.show()