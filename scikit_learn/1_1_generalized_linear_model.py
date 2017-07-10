# Scikit-Learn Linear Model
"""
from sklearn import linear_model

reg = linear_model.LinearRegression()
data =[[0, 0],
       [1, 1],
       [2, 2]]
label=[0,1,2]
reg.fit(X=data, y=label)
print reg.coef_
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model

# diabetes.data.shape ==> (442, 10)
diabetes = datasets.load_diabetes()

# use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_Y_train = diabetes.target[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_Y_test = diabetes.target[-20:]

# create linear regression object
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_Y_train)

print 'coefficients:', regr.coef_

# MSE
print 'mean squared error:%.2f' % np.mean((regr.predict(diabetes_X_test)-diabetes_Y_test)**2)
# Variance
print 'variance score:%.2f' % regr.score(diabetes_X_test, diabetes_Y_test)

# Plot
plt.scatter(diabetes_X_test, diabetes_Y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
