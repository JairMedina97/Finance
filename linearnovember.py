import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
  
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 
  
def main(): 
    # observations 
    x = np.array([48123,
                  73181,
                  56581,
                  45869,
                  71805,
                  69117,
                  74168,
                  62852,
                  82372,
                  52594,
                  56183,
                  77765,
                  52225,
                  62992,
                  54181,
                  58570,
                  56422,
                  48375,
                  46145,
                  56227,
                  80776,
                  77385,
                  54909,
                  68388,
                  43529,
                  53578,
                  53386,
                  59970,
                  58003,
                  73381,
                  80088,
                  46744,
                  64894,
                  52752,
                  61843,
                  54021,
                  50051,
                  60212,
                  59195,
                  63870,
                  50570,
                  56521,
                  51340,
                  59206,
                  68358,
                  57513,
                  71535,
                  70979,
                  43469,
                  59305,
                  60434]) 

    y = np.array([1038,
                  409,
                  1158,
                  234,
                  3446,
                  1241,
                  2597,
                  2407,
                  5121,
                  474,
                  743,
                  387,
                  1369,
                  1364,
                  1122,
                  1209,
                  958,
                  372,
                  219,
                  363,
                  3619,
                  4240,
                  2191,
                  1515,
                  308,
                  1183,
                  379,
                  592,
                  270,
                  1746,
                  1979,
                  3013,
                  1182,
                  1321,
                  683,
                  1031,
                  349,
                  1804,
                  1363,
                  1552,
                  459,
                  335,
                  684,
                  839,
                  1366,
                  603,
                  1110,
                  3140,
                  246,
                  1129,
                  500]) 
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    print("Estimated coefficients: \nb_0 = {}  \  \nb_1 = {}".format(b[0], b[1])) 
  
    # plotting regression line 
    plot_regression_line(x, y, b) 
  
if __name__ == "__main__": 
    main()



def func(x, a, b, c):
  return a * np.exp(-b * x) + c
  #return a * np.log(b * x) + c

#x = np.linspace(1,5,50)   # changed boundary conditions to avoid division by 0
x = np.array([48123,
                  73181,
                  56581,
                  45869,
                  71805,
                  69117,
                  74168,
                  62852,
                  82372,
                  52594,
                  56183,
                  77765,
                  52225,
                  62992,
                  54181,
                  58570,
                  56422,
                  48375,
                  46145,
                  56227,
                  80776,
                  77385,
                  54909,
                  68388,
                  43529,
                  53578,
                  53386,
                  59970,
                  58003,
                  73381,
                  80088,
                  46744,
                  64894,
                  52752,
                  61843,
                  54021,
                  50051,
                  60212,
                  59195,
                  63870,
                  50570,
                  56521,
                  51340,
                  59206,
                  68358,
                  57513,
                  71535,
                  70979,
                  43469,
                  59305,
                  60434])
#y = func(x, 2.5, 1.3, 0.5)
y = np.array([1038,
                  409,
                  1158,
                  234,
                  3446,
                  1241,
                  2597,
                  2407,
                  5121,
                  474,
                  743,
                  387,
                  1369,
                  1364,
                  1122,
                  1209,
                  958,
                  372,
                  219,
                  363,
                  3619,
                  4240,
                  2191,
                  1515,
                  308,
                  1183,
                  379,
                  592,
                  270,
                  1746,
                  1979,
                  3013,
                  1182,
                  1321,
                  683,
                  1031,
                  349,
                  1804,
                  1363,
                  1552,
                  459,
                  335,
                  684,
                  839,
                  1366,
                  603,
                  1110,
                  3140,
                  246,
                  1129,
                  500]) 
    
yn = y + 0.2*np.random.normal(size=len(x))

popt, pcov = curve_fit(func, x, yn)

plt.figure()
plt.plot(x, yn, 'ko', label="Original Noised Data")
plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.show()

