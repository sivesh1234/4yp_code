#Weather forecast timeseries RNN MULTIVARIATE


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

#if short > long then 'signal' = 1

vod['signal'][short_window:] = np.where(vod['short_mavg'][short_window:] > vod['long_mavg'][short_window:], 1.0, 0.0)




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


BATCH_SIZE = 100 #Batch size no. of periods fed intro training
BUFFER_SIZE = 10000 #Buffer size greater than sample size to ensure perfect shuffle
EVALUATION_INTERVAL = 200 #???
EPOCHS = 3 #No. of times model trains over full data set

 #Splits up the testing and training data
past_history = 90#periods fed into prediction for testing
future_target = 30#periods predicted for testing
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


 #Returns training data set = x and labels = y
 #dataset can be multivariate but target has to be univariate
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
 #Returns testing data set = x and labels = y
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

train_data_multi = (x_train_multi, y_train_multi)
val_data_multi = (x_val_multi, y_val_multi)

#Turns datasets into tf.data.Datasets ready for tensorflow
# train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
# #Shuffles and batches training data
# train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
#
# val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
# #Batches testing data
# val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

final_returns = []
total_returns = 0
plot_returns = []
multiple_returns = []

#Defines plotting for multistep prediction

# Over 3000 days this makes 100 trades, one every 30 days
def max_returns(history, true_future, prediction):
  #This randomly decides a buy,sell,hold
  signal_choice = [-1,0,1]
  signal = random.choice(signal_choice)
  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  start = history[89][1]
  end = true_future[29]
  if end > start:
      signal = 1
  else:
      signal = -1
  returns = end - start
  returns = returns*signal
  global final_returns
  global total_returns
  global plot_returns
  total_returns = total_returns + returns
  # print(total_returns)
  plot_returns.append(total_returns)

  # plt.plot(num_in, np.array(history[:, 1]), label='History')
  # plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
  #          label='Actual Price')
  # if prediction.any():
  #   plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
  #            label='Predicted Price')
  # plt.legend(loc='upper left')
  # plt.show()



#Plots an example with no prediction
# for x, y in train_data_multi.take(1):
#   multi_step_plot(x[0], y[0], np.array([0]))





# model = tf.keras.models.load_model('saved_model/my_model')



for alpha in range(0,3000,30):

    global total_returns
    global plot_returns
    global multiple_returns
    max_returns(x_val_multi[alpha],y_val_multi[alpha],y_val_multi[alpha])

plt.figure()
plt.plot(plot_returns)
plt.figure()
plt.plot(x_val_multi[:,1][0:3000,1])
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
