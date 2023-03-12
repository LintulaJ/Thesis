import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("equilib_data_squirrels.csv")
sns.set_theme()
sns.lineplot(x = "mu", y="R", data=df)
sns.lineplot(x = "mu", y="I", data=df)
plt.show()