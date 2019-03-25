import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output

data = pd.read_csv('../input/pokemon.csv')

print(data.index.name)
data.index.name="index_name"
data.head()

data.head()

data3 = data.copy()
data3.index = range(100,900,1)
data3.head()

data1 = data.set_index(["Type 1", "Type 2"])
data1.head(100)

dic = {"treatment" : ["A", "A", "B", "B"]}, "gender":["F", "M", "F", "M"], "response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame

df.pivot(index = "treatment", columns = "gender" , values = "response")

df1 = df.set_index(["treatment", "gender"])
df1.unstack(level=0)

df1.unstack(level=1)

df2.swaplevel(0,1)

pd.melt(df,id_vars="treatment", value_vars = ["age", "response"])

df.groupby("treatment").mean()
df.groupby("treatment").age.max()
df.groupby("treatment")[["age","response"]].min()
