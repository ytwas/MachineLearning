import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output

data = pd.read_csv('../input/pokemon.csv')
threshold = sum(data.Speed)/len(data.Speed)
data["speed_level"]= ["high" if i > threshold else "low" for i in data.Speed]
data.loc[:10, ["speed_level","Speed"]]