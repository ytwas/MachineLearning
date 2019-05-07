import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output

data = pd.read_csv('../input/pokemon.csv')

country = ["Spain", "France"]
population = ["11", "12"]
list_label = ["country", " population"]
list_col = [country, population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
#print(df)
df["capital"] = ["madrid", "paris"]
