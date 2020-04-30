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
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#Matplot lib stuff
tar_data = pdr.get_data_yahoo('TEF.MC',
                          start=datetime.datetime(2017, 11, 1),
                          end=datetime.datetime(2018, 12, 31))

B_data = pdr.get_data_yahoo('BT-A.L',
                          start=datetime.datetime(2017, 11, 1),
                          end=datetime.datetime(2017, 12, 31)) #year, month, date
C_data = pdr.get_data_yahoo('VOD',
                          start=datetime.datetime(2017, 11, 1),
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
def create_table(data1,data2,data3):
    d = {'stock_1':data1,'stock_2':data2,'stock_3':data3}
    table = pd.DataFrame(data=d)
    return table
# d = {'BT':B_data['Close'],'VOD':C_data['Close'],'TEF':tar_data['Close']}
# table = pd.DataFrame(data=d)

# table = tar_data
# table = table.drop(columns=['High','Low','Open','Close','Volume','Adj Close'])



# plt.figure(figsize=(14, 7))
# for c in returns.columns.values:
#     plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
# plt.legend(loc='upper right', fontsize=12)
# plt.ylabel('daily returns')


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(abs(x)) - 1})
    bound = (0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result





def calculate_weights(table):
    returns = table.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()


    risk_free_rate = 0.00

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)

    weight_allocation = max_sharpe['x']

    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T


    print("-"*80)
    print(max_sharpe_allocation)


    return weight_allocation

def print_cov(table):
    returns = table.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    print(cov_matrix)




# table = create_table(B_data['Close'],C_data['Close'],tar_data['Close'])
# plt.plot(table)
# plt.show()
# calculate_weights(table)
