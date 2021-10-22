# imports
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse
import seaborn as sns
pd.options.display.float_format = '{:.5f}'.format
import warnings
import math
import scipy.stats as stats
import scipy
from sklearn.preprocessing import scale
warnings.filterwarnings('ignore')


os.chdir("D:\Session 2021-22\Winter-21\PDS\Day 9")


data = pd.read_csv('LinRegression.csv')  # load data set

sns.pairplot(data)

#x = data.iloc[:, 0].values#.reshape(-1, 1)  # values converts it into a numpy array
#y = data.iloc[:, 1].values#.reshape(-1, 1)  

# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or 
#                                array.reshape(1, -1) if it contains a single sample

#print (x)
#print (y)


X=data["CGPA"]
Y=data["Percentage"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 333)
print('Training Data Count: {}'.format(X_train.shape[0]))
print('Testing Data Count: {}'.format(X_test.shape[0]))


X_train = sm.add_constant(X_train)
results = sm.OLS(y_train, X_train).fit()
results.summary()

X_test = sm.add_constant(X_test)

y_preds = results.predict(X_test)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
plt.figure(dpi = 75)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Actual Scores", fontdict=font)
plt.ylabel("Estimated Scores", fontdict=font)
plt.title("Model: Actual vs Estimated Scores", fontdict=font)
plt.show()



