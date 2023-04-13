import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


points = 1000
data = np.zeros((points*2, 3))
label = ["g(x)" for i in range(points*2)]

for i in range(points):
    data[i,0] = 0.1 + 0.5 * i/points
    data[i,1] = 0
    data[i,2] = -data[i,0]
    data[i+points,0] = 0.1 + 0.5 * i/points
    data[i+points,1] = max(0, (data[i+points,0]-0.25)/data[i+points,0])
    data[i+points,2] = -data[i,0]
    if data[i,0] < 0.25:
        label[i+points] = "stable"
        label[i] = "stable"
    else:
        label[i+points] = "stable"
        label[i] = "unstable"

df = pd.DataFrame(data, columns = ["r", "N", "c"])
df["equilibria"] = label

sns.set_theme()
sns.lineplot(x = "r", y="N", data=df, hue="equilibria")
plt.show()

sns.set_theme()
sns.lineplot(x = "r", y="c", data=df)
plt.show()