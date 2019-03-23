import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output

data = pd.read_csv('../input/pokemon.csv')

i = 0
while i != 5:
    #print('i is ', i)
    i += 1
#print(i, ' is equal to 5')

lis = range(1,6)
for i in lis:
    print(' i is ', i )
print('')

for index, value in enumerate(lis):
    print(index," : ",value)
print('')

dictionary = {'spain': 'madrid', 'france': 'paris'}
for key, value in dictionary.items():
    print(key," : ",value)
print('')

for index,value in data [['Attack']][0:1].iterrows():
    print(index," : ",value)
