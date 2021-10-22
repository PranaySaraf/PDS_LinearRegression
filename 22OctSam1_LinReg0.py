# imports
import numpy as np
import matplotlib.pyplot as plt
#Run using Cntr+Enter


# from numpy.random import seed
# from numpy.random import randint 
# import random
# random.seed(1)
# x = randint(0, 30, 10)
# y = 15 + 3 * x + randint(1, 20, 10)

# select all and press the ctrl + / for multiple line comments

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 4, 2, 4, 5])

print (x)
print (y)


# plot
plt.scatter(x,y,s=100, color='magenta')
plt.xlabel('x', fontsize = 20, color='red')
plt.ylabel('y', fontsize = 20, color='green')
plt.show()
#plt.plot(x,y, color='red')
#plt.show()

from sklearn.linear_model import LinearRegression
X=x.reshape(-1, 1)
print (X)
print (y)
#model = LinearRegression()
#model.fit(X, y)
model = LinearRegression().fit(X, y)
print (model)


print('intercept: b =', model.intercept_)
print('slope: m =', model.coef_)

y_pred1 = model.intercept_ + model.coef_ * X
print(x.reshape(1,-1))
print(y)
print(y_pred1)






