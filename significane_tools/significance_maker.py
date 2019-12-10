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
#Matplot lib stuff
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
#Getting data
vod = pdr.get_data_yahoo('VOD',
                          start=datetime.datetime(2000, 10, 26),
                          end=datetime.datetime(2019, 10, 26)) #year, month, date
short_window = 41
long_window = 101

#gives signals the same index (dates) as vod
vod['signal'] = 0.0
vod['short_mavg'] = vod['Close'].rolling(window=short_window,
                                              min_periods=1,center=False).mean()
vod['long_mavg'] = vod['Close'].rolling(window=long_window,min_periods=1,center=False).mean()
vod['short_mavg'] = vod['short_mavg'].shift(periods=(-50))
vod['long_mavg'] = vod['long_mavg'].shift(periods=(-50))
vod['signal'][short_window:] = np.where(vod['short_mavg'][short_window:] > vod['long_mavg'][short_window:], 1.0, 0.0)

last3000 = vod['Close'][-3000:]
open_price = vod['Close'][0]

TRAIN_SPLIT = 1500


df = vod

features_considered = ['signal','Close']
features = df[features_considered]
features.index = df.index
# features.plot(subplots=True)
# features.plot()
dataset = features.values
print(len(dataset))

# data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
# data_std = dataset[:TRAIN_SPLIT].std(axis=0)
# dataset = (dataset-data_mean)/data_std


#Creates a data set with associated labels. Takes index for these data sets as inputs

#Think of it as two parallel vertical arrays, one of width history_size and one of width target_size

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)




 #Splits up the testing and training data
past_history = 90 #periods fed into prediction for testing
future_target = 30 #periods predicted for testing
STEP = 1 #How many periods it samples over



def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

#creates an array from -length to zero
def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps




 #MULTISTEP MODEL



 #Returns testing data set = x and labels = y
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)




final_returns = []
total_returns = 0
plot_returns = []
multiple_returns = []

#Defines plotting for multistep prediction
open_price = x_val_multi[0][0][1]
# Over 3000 days this makes 100 trades, one every 30 days
def get_returns(history, true_future, prediction):
  #This randomly decides a buy,sell,hold
  signal_choice = [-1,0,1]
  signal = random.choice(signal_choice)

  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  start = history[89][1]
  end = true_future[29]
  returns = ((end - start)/open_price)*100
  returns = returns*signal
  global final_returns
  global total_returns
  global plot_returns
  total_returns = total_returns + returns
  plot_returns.append(total_returns)



#Monte Carlo simulation
no_simulations = 1000

for stepz in range(1,no_simulations):


    for alpha in range(0,3000,30):

        global total_returns
        global plot_returns
        global multiple_returns
        get_returns(x_val_multi[alpha],y_val_multi[alpha],y_val_multi[alpha])

    global final_returns
    global multiple_returns
    multiple_returns.append(plot_returns)
    final_returns.append(total_returns)
    total_returns = 0
    plot_returns = []


multiple_returns = np.array(multiple_returns)
multiple_returns = np.transpose(multiple_returns)
print(multiple_returns.shape)
print(len(final_returns))
final_returns = np.array(final_returns)
# print("MAX: {} MIN: {} MEAN: {} STD: {}".format(np.max(final_returns),np.amin(final_returns),np.mean(final_returns),np.std(final_returns)))

max = np.max(final_returns)
min = np.amin(final_returns)
mean = np.mean(final_returns)
std = np.std(final_returns)





plt.hist(final_returns,bins=20)
plt.show()
plt.figure()

plt.title('Monte-Carlo returns: MAX: {} MIN: {} MEAN: {} STD: {}'.format(np.max(final_returns),np.amin(final_returns),np.mean(final_returns),np.std(final_returns)))
plt.xlabel('Trades')
plt.ylabel('% of start price - returns ')
plt.plot(multiple_returns)


plt.show()


# plt.figure()
# plt.plot(y_val_multi, label='true')
# plt.plot(pred, label='pred')
# plt.legend()
# plt.show()

#.take(x) Return elements along axis x
#Takes 5 rows of validation data set, x is the input and y are the labels
# for x, y in val_data_multi.take(5):
#   multi_step_plot(x[0], y[0], model.predict(x)[0])
