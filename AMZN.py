# Import libraries
import pandas as pd
import numpy as np
import yfinance as yf

# Import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline

# Fetch the Amazon stock data
data = yf.download('AMZN', '2015-01-01')

# Visualise the data
data['Close'].plot(figsize=(15,7))
plt.ylabel('Close Price')
plt.title('Amazon Close Price')
plt.show()
