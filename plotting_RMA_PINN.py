import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("_RMA_equilib.csv")
df2 = df[df.N < 2]
#df = df["N" > 0.5]
sns.set_theme()
sns.lineplot(x = "N", y="P", data=df2, hue="equilibria")
plt.show()
