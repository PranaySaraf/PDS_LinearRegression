# imports
import os
import numpy as np
import pandas as pd  # To read data from csv
import matplotlib.pyplot as plt


os.chdir("D:\Session 2021-22\Winter-21\PDS\Day 9")

data = pd.read_csv('LinRegression.csv')  # load data set
x = data.iloc[:, 0].values#.reshape(-1, 1)  # values converts it into a numpy array
y = data.iloc[:, 1].values#.reshape(-1, 1)  

# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or 
#                                array.reshape(1, -1) if it contains a single sample

print (x)
print (y)


# plot
plt.scatter(x,y,s=10, color='red')
plt.xlabel('CGPA', fontsize = 20)
plt.ylabel('Percentage', fontsize = 20)
plt.show()
#plt.plot(x,y, color='red')
#plt.show()