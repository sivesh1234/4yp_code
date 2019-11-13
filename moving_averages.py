import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance #import yahoo finance

aapl = pdr.get_data_yahoo('AAPL',
                          start=datetime.datetime(2016, 10, 26),
                          end=datetime.datetime(2019, 10, 26)) #year, month, date
# Plot the closing prices for `aapl`
aapl['Close'].plot(grid=True)

# Show the plot
plt.show()
# Assign `Adj Close` to `daily_close`
daily_close = aapl[['Adj Close']] #Adjusted for dividends etc..
daily_pct_c = daily_close.pct_change() #percentage change of daily close
daily_pct_c.fillna(0, inplace=True)
daily_log_returns = np.log(daily_close.pct_change()+1)

#Moving avergaes

aapl['40'] = daily_close.rolling(window=40).mean()
aapl['252'] = daily_close.rolling(window=252).mean()
aapl[['Adj Close','40','252']].plot()
plt.show()
