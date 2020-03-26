from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import datetime
from pandas_datareader import data as pdr
import yfinance
import scipy.optimize as sco
#Matplot lib stuff
tar_data = pdr.get_data_yahoo('TEF.MC',
                          start=datetime.datetime(2009, 1, 1),
                          end=datetime.datetime(2017, 12, 31))

B_data = pdr.get_data_yahoo('BT-A.L',
                          start=datetime.datetime(2009, 1, 1),
                          end=datetime.datetime(2017, 12, 31)) #year, month, date
C_data = pdr.get_data_yahoo('VOD',
                          start=datetime.datetime(2009, 1, 1),
                          end=datetime.datetime(2017, 12, 31))
len_tar = len(tar_data['Close'])
len_B = len(B_data['Close'])
len_C = len(C_data['Close'])

min_len = min([len_tar,len_B,len_C])
dif_tar = len_tar - min_len
dif_B = len_B - min_len
dif_C = len_C - min_len

tar_data = tar_data[dif_tar:]
B_data = B_data[dif_B:]
C_data = C_data[dif_C:]

C_data.index = tar_data.index
B_data.index = tar_data.index


#gives signals the same index (dates) as tar_data
tar_data['BT_Close'] = B_data['Close']
tar_data['VOD_Close'] = C_data['Close']
tar_data['TEF_Close'] = tar_data['Close']

table = tar_data
table = table.drop(columns=['High','Low','Open','Close','Volume','Adj Close'])
plt.figure(figsize=(14, 7))
plt.subplot(211)
plt.plot(table.index, table["BT_Close"], lw=3, alpha=0.8)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in $')
plt.subplot(212)
plt.plot(table.index, table["VOD_Close"], lw=3, alpha=0.8)
plt.plot(table.index, table["TEF_Close"], lw=3, alpha=0.8)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in $')

returns = table.pct_change()

plt.figure(figsize=(14, 7))
for c in returns.columns.values:
    plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper right', fontsize=12)
plt.ylabel('daily returns')


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in xrange(num_portfolios):
        ###Produces negative weights as well
        # weights = 4*np.random.random(3)-2
        ###Produce 0-1 weights
        weights = np.random.random(3)
        weights /= np.sum(np.absolute(weights))
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 10000
risk_free_rate = 0.00



def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='g',s=300, label='Max Sharpe')
    # plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('BT, TEF and VOD Portfolio Optimization - Efficient Frontier')
    plt.xlabel('volatility - (annualised)')
    plt.ylabel('returns - (annualised)')
    plt.legend(labelspacing=0.8)

display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)



plt.show()
