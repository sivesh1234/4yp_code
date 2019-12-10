import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance #import yahoo finance




ticker = ""

np.set_printoptions(threshold=np.inf)
vod = pdr.get_data_yahoo('VOD',
                          start=datetime.datetime(2000, 5, 26),
                          end=datetime.datetime(2019, 10, 26))
                          #year, month, date
vod = vod.tail(3300)
short_window = 41
long_window = 101

signals = pd.DataFrame(index=vod.index) #gives signals the same index (dates) as vod
signals['signal'] = 0.0
signals['short_mavg'] = vod['Close'].rolling(window=short_window,
                                              min_periods=1,center=False).mean()

signals['long_mavg'] = vod['Close'].rolling(window=long_window,min_periods=1,center=False).mean()
# signals['short_mavg'] = signals['short_mavg'].shift(periods=(-50))
# signals['long_mavg'] = signals['long_mavg'].shift(periods=(-50))

#if short > long then 'signal' = 1
#unsure what short_window does here***************
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
#if 0 --> 1 position = 1 if 1--> 0 position = -1
signals['positions'] = signals['signal'].diff()
#Plot buy and sells
fig = plt.figure()
yaxis = fig.add_subplot(111,ylabel='Price')
vod['Close'].plot(ax=yaxis,color='r')
signals[['short_mavg','long_mavg']].plot(ax=yaxis)
yaxis.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0],'^', markersize=10, color='b')
yaxis.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0], 'v', markersize=10, color='r')

#Backtesting the strategy
initial_capital = float(0.0)
positions = pd.DataFrame(index=signals.index).fillna(0.0)
positions['vod'] = 1*signals['signal']
portfolio = positions.multiply(vod['Adj Close'], axis=0)

pos_diff = positions.diff()
portfolio['holdings'] = (positions.multiply(vod['Adj Close'], axis=0)).sum(axis=1)

# Create Portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(vod['Adj Close'], axis=0)).sum(axis=1).cumsum()
# Add `total` to portfolio
open_price = vod['Close'][0]

portfolio['total'] = (((portfolio['cash'] + portfolio['holdings'])/open_price)*100)
# portfolio['total'] = (portfolio['cash'] + portfolio['holdings'])
# Add `returns` to portfolio

portfolio['returns'] = portfolio['total'].pct_change()
fig = plt.figure()



# Plot the portfolio curve
ax1 = fig.add_subplot(111, ylabel='PnL %')
portfolio['total'].plot(ax=ax1, lw=2.)

fig2 = plt.figure()

vod['Close'].plot(grid=True)


plt.show()
