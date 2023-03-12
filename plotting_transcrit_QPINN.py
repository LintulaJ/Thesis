import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("data_bifurc_transcrit.csv")
sns.set_theme()
sns.lineplot(x = "r", y="c", data=df)
plt.show()

df = pd.read_csv("data_equilib_transvirt.csv")
sns.set_theme()
sns.lineplot(x = "r", y="N", hue="equilibria", data=df)
plt.show()