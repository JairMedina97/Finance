def clear():
    print('\n' * 40)
import math
import statistics
import datetime 
import numpy as np
import scipy as sp
import array as arr
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as dates
from datetime import timedelta, date
import matplotlib.image as mpimg

plt.style.use('dark_background')

x= range(0,501)
data= pd.read_csv("SP500_17-19.csv")

sp500 = np.asarray(data)
mean = sum(sp500) / len(sp500)

variance = np.var(sp500)
strddev = math.sqrt(variance)

mas = mean + strddev
menos = mean - strddev
s="Standard Deviation"
   
plt.plot(x,sp500, color='green')
plt.title("S&P 500 (2017 - 2019)", fontsize=18, y=1.02)
plt.xlabel("Date", fontsize=14, labelpad=15)
plt.ylabel("Index", fontsize=14, labelpad=15)
plt.axhline(y=mas, color='yellow', linestyle='--', linewidth=1.5)
plt.axhline(y=menos, color='yellow', linestyle='--', linewidth=1.5)
plt.axvline(x=130, color='red', linestyle='--', linewidth=1.5)
plt.axvline(x=303, color='red', linestyle='--', linewidth=1.5)
plt.text(500, y=mas,s="Standard Deviation", fontsize=14,
         bbox=dict(facecolor='Black', boxstyle="round, pad=0.4"));
plt.text(500, y=menos,s="Standard Deviation", fontsize=14,
         bbox=dict(facecolor='Black', boxstyle="round, pad=0.4"));
plt.grid(b=True, linewidth=0.5);
plt.show()

#img=mpimg.imread('SP500.png')
#imgplot = plt.imshow(img)
plt.show()


            
