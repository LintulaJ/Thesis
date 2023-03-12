import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

z = "R"

df = pd.read_csv("_sqrl_sim_3.csv")
sns.set_style("darkgrid")
sns_plot = plt.axes(projection="3d")

sns_plot.scatter3D(df["t"], df["mu"], df[z], c=df[z])
sns_plot.set_xlabel("t")
sns_plot.set_ylabel("mu")
sns_plot.set_zlabel(z)

plt.show()