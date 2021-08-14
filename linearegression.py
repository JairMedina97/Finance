import math
import numpy as np
from sklearn.linear_model import LinearRegression

x1 = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y1 = np.array ([5, 20, 14, 32, 22, 38])

model = LinearRegression()
model.fit(x1,y1) 

#model = LinearRegression().fit(x,y)

r_sq = model.score(x1,y1)
print('Coefficient of determination: ', r_sq)
print('Intercept: ', model.intercept_)
print('slope: ', model.coef_)
y_pred1 = model.predict(x1) # make predictions
print('predict response: ', y_pred1, sep='\n')

import matplotlib.pyplot as plt

x_new = np.arange(5).reshape((-1, 1))
y_new = model.predict(x_new)

plt.scatter(x1,y1)
plt.plot(x1,y_pred1, color='red')
plt.show()

#-------------------------------------------------------------------------------
# Multiple Linear Regression


x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55,34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np. array(y)

model = LinearRegression().fit(x,y)
r_sq = model.score(x,y)
print('coefficient of determination: ', r_sq)
print('intercept: ', model.intercept_)
print('slope: ', model.coef_)
y_pred = model.predict(x)
print('predicted response: ', y_pred, sep='\n')



#-------------------------------------------------------------------------------
# Polynomial Regression
#from sklearn.preprocessing import PolynomialFeatures

#x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
#y = np.array([15, 11, 2, 8, 25, 32])

#transformer = PolynomialFeatures(degree=2, include_bias=False)
#transformer.fit(x)

#x_ = transformer.transform(x)

#x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

#print(x_)

#model = LinearRegression().fit(x_,y)
#r_sq = model.score(x_, y)
#print('coefficient of determination:', r_sq)
#print('intercept:', model.intercept_)
#print('coefficients:', model.coef_)
#print(x_)
#r_sq = model.score(x_, y)
#print('coefficient of determination:', r_sq)
#print('intercept:', model.intercept_)
#print('coefficients:', model.coef_)

#y_pred = model.predict(x_)
#print('predicted response: ', y_pred, sep='\n')


#-------------------------------------------------------------------------------
# Instructivo

# Step 1: Import packages
#import numpy as np
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures

# Step 2a: Provide data
#x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
#y = [4, 5, 20, 14, 32, 22, 38, 43]
#x, y = np.array(x), np.array(y)

# Step 2b: Transform input data
#x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

# Step 3: Create a model and fit it
#model = LinearRegression().fit(x_, y)

# Step 4: Get results
#r_sq = model.score(x_, y)
#intercept, coefficients = model.intercept_, model.coef_

# Step 5: Predict
#y_pred = model.predict(x_)

#-------------------------------------------------------------------------------
#Advanced Linear Regression statsmodels
#import statsmodels.api as sm
#x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
#y = [4, 5, 20, 14, 32, 22, 38, 43]
#x, y = np.array(x), np.array(y)
#x = sm.add_constant(x)
#model = sm.OLS(y, x)
#results = model.fit()

#print(results.summary())
#print('coefficient of determination:', results.rsquared)
#print('adjusted coefficient of determination:', results.rsquared_adj)
#print('regression coefficients:', results.params)
#print('predicted response:', results.fittedvalues, sep='\n')
#print('predicted response:', results.predict(x), sep='\n')



#-------------------------------------------------------------------------------
#Ejemplo básico desde Pandas #Archivo data no existe 

#import numpy as np
#import matplotlib.pyplot as plt  # To visualize
#import pandas as pd  # To read data
#from sklearn.linear_model import LinearRegression

#data = pd.read_csv('data.csv')  # load data set
#X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
#Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
#linear_regressor = LinearRegression()  # create object for the class
#linear_regressor.fit(X, Y)  # perform linear regression
#Y_pred = linear_regressor.predict(X)  # make predictions

#plt.scatter(X, Y)
#plt.plot(X, Y_pred, color='red')
#plt.show()

#scikit-learn if you don’t need detailed results and want to use the approach consistent with other regression techniques
#statsmodels if you need the advanced statistical parameters of a model






