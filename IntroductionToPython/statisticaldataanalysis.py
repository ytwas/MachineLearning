import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output

data = pd.read_csv('../input/pokemon.csv')

#time_list = ["1992-03-08", "1992-04-12"]
#print(type(time_list[1]))

#datatime_object = pd.to_datetime(time_list)

#print(datatime_object)

#import warnings.filterwarnings("ignore")

#data2 = data.head()
#date_list = {"1992-01-10", "1992-02-10","1992-03-10","1993-03-15","1993-03-16"}
#datetime_object = pd.to_datetime(date_list)
#data2["date"] = datetime_object
#data2 = data2.set_index("date")
print(data["HP"])
#print(data[["HP"]])