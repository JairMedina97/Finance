import pandas_datareader as pdr
import datetime
import numpy as np
name    = input("Enter stock ticker : ")
stock = pdr.get_data_yahoo(name, 
                          start=datetime.datetime(2008, 1, 1), 
                          end=datetime.datetime(2019, 9, 2))
import pandas as pd
#aapl.to_csv('data/aapl_ohlc.csv')
#df = pd.read_csv('data/aapl_ohlc.csv', header=0, index_col='Date', parse_dates=True)

# Import Matplotlib's `pyplot` module as `plt`
import matplotlib.pyplot as plt

# Plot the closing prices for `aapl`
stock['Close'].plot(grid=True)

# Show the plot
plt.show()

# Isolate the adjusted closing prices 
adj_close_px = stock['Close']

# Short moving window rolling mean
stock['42'] = adj_close_px.rolling(window=40).mean()

# Long moving window rolling mean
stock['252'] = adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
#stock[['Adj Close', '42', '252']].plot()

# Show plot
#plt.show()

# Isolate the `Adj Close` values and transform the DataFrame
daily_close_px = stock[['Adj Close']].reset_index().pivot('Date', 'Adj Close')

# Calculate the daily percentage change for `daily_close_px`
daily_pct_change = stock['Close'].pct_change()

# Show the resulting plot
#daily_pct_change.plot()

# Plot the distributions
daily_pct_change.hist(bins=50)
plt.show()

# Define the minumum of periods to consider 
min_periods = 75 

# Calculate the volatility
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods) 

# Plot the volatility
vol.plot(figsize=(8, 8))

# Show the plot
plt.show()

##
#---------------------------------------------------------------

# Calculate the cumulative daily returns
#cum_daily_return = (1 + daily_pct_change).cumprod()

# Plot the cumulative daily returns
#cum_daily_return.plot(figsize=(12,8))

####



#-----------------------------------------------------------------------------
## Quant trading

# Initialize the short and long windows
short_window = 40
long_window = 100

#Initialize 'signals' DataFrame with the 'signal' column
signals = pd.DataFrame(index=stock.index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = stock['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = stock['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)

# Generate trading orders
signals['positions'] = signals['signal'].diff()

# Print `signals`
print(signals)

# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')

# Plot the closing price
stock['Close'].plot(ax=ax1, color='blue', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=15, color='g')
         
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=15, color='r')
         
# Show the plot
plt.show()

##----------------------------------------------------------------------------
# Set the initial capital
initial_capital= float(10000.0)
# Create a DataFrame `positions`
positions = pd.DataFrame(index=signals.index).fillna(0.0)

# Buy a 100 shares
positions[name] = 100*signals['signal'] 

# Initialize the portfolio with value owned   
portfolio = positions.multiply(stock['Adj Close'], axis=0)

# Store the difference in shares owned 
pos_diff = positions.diff()

# Add `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(stock['Adj Close'], axis=0)).sum(axis=1)

# Add `cash` to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(stock['Adj Close'], axis=0)).sum(axis=1).cumsum() 

# Add `total` to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# Print the first lines of `portfolio`
print(portfolio)

# Create a figure
fig = plt.figure()

ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2.)

ax1.plot(portfolio.loc[signals.positions == 1.0].index, 
         portfolio.total[signals.positions == 1.0],
         '^', markersize=10, color='g')
ax1.plot(portfolio.loc[signals.positions == -1.0].index, 
         portfolio.total[signals.positions == -1.0],
         'v', markersize=10, color='r')

# Show the plot
plt.show()

# Isolate the returns of your strategy
returns = portfolio['returns']

# annualized Sharpe ratio
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

# Print the Sharpe ratio
print(sharpe_ratio)

# Define a trailing 252 trading day window
window = 252

# Calculate the max drawdown in the past window days for each day 
rolling_max = stock['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = stock['Adj Close']/rolling_max - 1.0

# Calculate the minimum (negative) daily drawdown
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

# Plot the results
daily_drawdown.plot()
max_daily_drawdown.plot()

# Show the plot
plt.show()

# Get the number of days in `stock`
days = (stock.index[-1] - stock.index[0]).days

# Calculate the CAGR 
cagr = ((((stock['Adj Close'][-1]) / stock['Adj Close'][1])) ** (365.0/days)) - 1

# Print the CAGR
print(cagr)

















#--------------------------------------------------------------------------

# Plot two charts to assess trades and equity curve
#fig = plt.figure()
#fig.patch.set_facecolor('white')     # Set the outer colour to white
#ax1 = fig.add_subplot(211,  ylabel='Price in $')

# Plot the AAPL closing price overlaid with the moving averages
#stock['Adj Close'].plot(ax=ax1, color='r', lw=2.)
#signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the "buy" trades against AAPL
#ax1.plot(signals.loc[signals.positions == 1.0].index, 
#             signals.short_mavg[signals.positions == 1.0],
#             '^', markersize=10, color='r')

# Plot the "sell" trades against AAPL
#ax1.plot(signals.loc[signals.positions == -1.0].index, 
#             signals.short_mavg[signals.positions == -1.0],
#             'v', markersize=10, color='g')

# Plot the equity curve in dollars
#ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
#returns['total'].plot(ax=ax2, lw=2.)

# Plot the "buy" and "sell" trades against the equity curve
#ax2.plot(returns.loc[signals.positions == 1.0].index, 
#             returns.total[signals.positions == 1.0],
#             '^', markersize=10, color='r')
#ax2.plot(returns.loc[signals.positions == -1.0].index, 
#             returns.total[signals.positions == -1.0],
#             'v', markersize=10, color='g')

    # Plot the figure
#fig.show()












