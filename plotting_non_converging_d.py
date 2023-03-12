
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

points = 10000
data = np.zeros((points*2, 2))
label = ["g(x)" for i in range(points*2)]

for i in range(points):
    data[i,0] = i/points
    data[i,1] = i/points
    label[i] = "g(x)"

for i in range(points):
    data[i+points,0] = i/points
    data[i+points,1] = i/points + 0.01 * np.sin(100*i/points)
    label[i+points] = "f(x)"

df = pd.DataFrame(data, columns = ["x", "y"])
df["function"] = label

sns.set_theme()
sns.lineplot(x = "x", y="y", hue="function", data=df)
plt.show()