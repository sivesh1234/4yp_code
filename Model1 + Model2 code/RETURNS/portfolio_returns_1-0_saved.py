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
from efficient_frontier_module import *

# save numpy array as npy file
from numpy import asarray
from numpy import save

from numpy import load


#Matplot lib stuff
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
#Getting data

#BT, VOD AND TEF is order for working
tar_data = pdr.get_data_yahoo('BT-A.L',
                          start=datetime.datetime(1990, 10, 26),
                          end=datetime.datetime(2019, 10, 26))

B_data = pdr.get_data_yahoo('VOD',
                          start=datetime.datetime(1990, 10, 26),
                          end=datetime.datetime(2019, 10, 26)) #year, month, date
C_data = pdr.get_data_yahoo('TEF.MC',
                          start=datetime.datetime(1990, 10, 26),
                          end=datetime.datetime(2019, 10, 26))                          #year, month, date

short_window = 30
long_window = 80

#Make all dataframes same length
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
tar_data['A_Close'] = tar_data['Close']
tar_data['B_Close'] = B_data['Close']
tar_data['C_Close'] = C_data['Close']



tar_data['A_signal'] = 0.0
tar_data['B_signal'] = 0.0
tar_data['C_signal'] = 0.0
tar_data['A_short_mavg'] = tar_data['A_Close'].rolling(window=short_window,
                                              min_periods=1,center=False).mean()

tar_data['A_long_mavg'] = tar_data['A_Close'].rolling(window=long_window,min_periods=1,center=False).mean()

tar_data['A_signal'][short_window:] = np.where(tar_data['A_short_mavg'][short_window:] > tar_data['A_long_mavg'][short_window:], 1.0, 0.0)
tar_data['B_short_mavg'] = tar_data['B_Close'].rolling(window=short_window,
                                              min_periods=1,center=False).mean()

tar_data['B_long_mavg'] = tar_data['B_Close'].rolling(window=long_window,min_periods=1,center=False).mean()

tar_data['B_signal'][short_window:] = np.where(tar_data['B_short_mavg'][short_window:] > tar_data['B_long_mavg'][short_window:], 1.0, 0.0)
tar_data['C_short_mavg'] = tar_data['C_Close'].rolling(window=short_window,
                                              min_periods=1,center=False).mean()

tar_data['C_long_mavg'] = tar_data['C_Close'].rolling(window=long_window,min_periods=1,center=False).mean()

tar_data['C_signal'][short_window:] = np.where(tar_data['C_short_mavg'][short_window:] > tar_data['C_long_mavg'][short_window:], 1.0, 0.0)



TRAIN_SPLIT = 1500


df = tar_data

A_features_considered = ['A_signal','A_Close','B_Close','C_Close']
B_features_considered = ['B_signal','B_Close','C_Close','A_Close']
C_features_considered = ['C_signal','C_Close','A_Close','B_Close']


A_features = df[A_features_considered]
A_features.index = df.index
A_dataset = A_features.values

B_features = df[B_features_considered]
B_features.index = df.index
B_dataset = B_features.values

C_features = df[C_features_considered]
C_features.index = df.index
C_dataset = C_features.values

# data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
# data_std = dataset[:TRAIN_SPLIT].std(axis=0)
# dataset = (dataset-data_mean)/data_std


#Creates a data set with associated labels. Takes index for these data sets as inputs

#Think of it as two parallel vertical arrays, one of width history_size and one of width target_size

def multivariate_data(dataset, target,  start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []
  full_y = []
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
    if single_step:
      full_y.append(dataset[i+target_size])
    else:
      full_y.append(dataset[i:i+target_size])


  return np.array(data), np.array(labels), np.array(full_y)


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
# x_train_multi, y_train_multi, x_val_all = multivariate_data(dataset, dataset[:, 0], 0,
#                                                  TRAIN_SPLIT, past_history,
#                                                  future_target, STEP)
 #Returns testing data set = x and labels = y
A_x_val_multi, A_y_val_multi, A_y_val_all = multivariate_data(A_dataset, A_dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)
B_x_val_multi, B_y_val_multi, B_y_val_all = multivariate_data(B_dataset, B_dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)
C_x_val_multi, C_y_val_multi, C_y_val_all = multivariate_data(C_dataset, C_dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)


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
A_signals = []
B_signals = []
C_signals = []

#Defines plotting for multistep prediction
A_open_price = A_x_val_multi[0][0][1]
B_open_price = B_x_val_multi[0][0][1]
C_open_price = C_x_val_multi[0][0][1]

# Over 3000 days this makes 100 trades, one every 30 days
def predicted_weighted_returns(history, true_future, prediction ,weight,open_price):
  #This randomly decides a buy,sell,hold

  signal = 0
  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  start = history[89][1]
  start_10 = history[89][0]

  std = np.std(history[:][1])
  end = true_future[29][1]


  prediction_average = prediction


  #This sets the signal to the best option
  print("predicted_average {}".format(prediction_average))
  if prediction_average > 0.98:
      signal = 1
      print("long")
  elif prediction_average < 0.02:
      signal = -1
      print("short")
  else:
      signal = 0
      print("flat")
  returns = ((end - start)/open_price)*100
  # if weight < -0.5:
  #     signal = 1
  # elif weight > 0.5:
  #     signal = 1
  # else:
  #     signal = signal
  returns = returns*signal*weight
  global final_returns
  global total_returns
  global plot_returns

  total_returns = total_returns + returns
  print("returns on trade {}".format(returns))
  # print(total_returns)
  # plot_returns.append(total_returns)
  # A_signals.append(signal)
  return signal, prediction_average
# def predicted_returns_B(history, true_future, prediction, weight_B):
#   #This randomly decides a buy,sell,hold
#
#   signal = 0
#   num_in = create_time_steps(len(history))
#   num_out = len(true_future)
#   start = history[89][3]
#   start_10 = history[89][2]
#
#   std = np.std(history[:][3])
#   end = true_future[29][3]
#
#   start_pred = prediction[2]
#   difference = start_10 - start_pred
#   prediction = prediction + difference
#   prediction_average = np.mean(prediction)
#   predicted_end = prediction[29]
#   #This sets the signal to the best option
#   print("predicted_average {}".format(prediction_average))
#   if prediction_average > 1.02:
#       signal = 1
#       print("long")
#   elif prediction_average < 0.02:
#       signal = -1
#       print("short")
#   else:
#       signal = 0
#   returns = ((end - start)/open_price)*100
#   returns = returns*signal*weight_B
#   global final_returns
#   global total_returns
#   global plot_returns
#   global B_signals
#   total_returns = total_returns + returns
#   print(total_returns)
#   # print(total_returns)
#   plot_returns.append(total_returns)
#   B_signals.append(signal)
#   return signal, returns
# def predicted_returns_C(history, true_future, prediction,weight_C):
#   #This randomly decides a buy,sell,hold
#
#   signal = 0
#   num_in = create_time_steps(len(history))
#   num_out = len(true_future)
#   start = history[89][5]
#   start_10 = history[89][4]
#
#   std = np.std(history[:][5])
#   end = true_future[29][5]
#
#   start_pred = prediction[4]
#   difference = start_10 - start_pred
#   prediction = prediction + difference
#   prediction_average = np.mean(prediction)
#   predicted_end = prediction[29]
#   #This sets the signal to the best option
#   print("predicted_average {}".format(prediction_average))
#   if prediction_average > 1.02:
#       signal = 1
#       print("long")
#   elif prediction_average < 0.02:
#       signal = -1
#       print("short")
#   else:
#       signal = 0
#   returns = ((end - start)/open_price)*100
#   returns = returns*signal*weight_C
#   global final_returns
#   global total_returns
#   global plot_returns
#   global C_signals
#   total_returns = total_returns + returns
#   print(total_returns)
#   # print(total_returns)
#   plot_returns.append(total_returns)
#   C_signals.append(signal)
#   return signal, returns
#







def multi_step_plot(history, true_future, prediction,signal=0,returns=0,trade=0):


###Initialise main figure
  plt.figure(0)

##First plot shows moving average position prediction
  fig1 = plt.subplot(411)
  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  start = history[89][0]
  start_pred = prediction[0]
  difference = start - start_pred
  prediction = prediction + difference
  plt.plot(num_in, np.array(history[:,0]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future[:,0]), 'bo',
           label='True Future')

###Set title to position
  if signal == 1:
      plt.title("LONG")
  elif signal == -1:
      plt.title("SHORT")
  else:
      plt.title("FLAT")
###Set succesful returns colour
  if returns >= 0:
      return_colour = "go"
  else:
      return_colour = "ro"



  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), return_colour,
             label='Predicted Future')
  fig1.set_ylim([-0.2,1.2])
  plt.legend(loc='upper left')





  plt.subplot(412)
  plt.plot(num_in, np.array(history[:,1]), label='BT History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future[:,1]), 'bo',
           label='True Future')

  plt.legend(loc='upper left')
  plt.subplot(413)
  plt.plot(num_in, np.array(history[:,2]), label='TEF History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future[:,2]), 'bo',
           label='True Future')

  plt.legend(loc='upper left')
  plt.subplot(414)
  plt.plot(num_in, np.array(history[:,3]), label='VOD History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future[:,3]), 'bo',
           label='True Future')

  plt.legend(loc='upper left')


  ###DISPLAYS PLOTS

  plt.show(block=False)
  plt.pause(0.5)
  plt.close()





A_predictions = load('a_predictions.npy')
B_predictions = load('b_predictions.npy')
C_predictions = load('c_predictions.npy')
weights_array = []
for alpha in range(0,3000,30):
    trade = int(alpha/30)
    print("-"*500)
    print("period {}".format(alpha/30))


    global total_returns
    global plot_returns
    global multiple_returns
    A_more_prices = []



    A_prices = np.array(A_x_val_multi[alpha][:,1])
    B_prices = np.array(B_x_val_multi[alpha][:,1])
    C_prices = np.array(C_x_val_multi[alpha][:,1])
    A_more_prices.append(A_prices)
    A_more_prices = np.array(A_more_prices)
    table = create_table(A_prices, B_prices, C_prices)
    plt.plot(A_prices)
    plt.show()

    weight_allocation = calculate_weights(table)
    print("weight allocation is {}".format(weight_allocation))

    # weight_A = (weight_allocation[0]/100)
    #
    # weight_B = (weight_allocation[1]/100)
    #
    # weight_C = (weight_allocation[2]/100)
    # weights_array.append(weight_allocation)
    weight_A = 1/3
    weight_B = 1/3
    weight_C = 1/3

    print("-"*80)
    print("trade1")
    A_signal,A_predict = predicted_weighted_returns(A_x_val_multi[alpha],A_y_val_all[alpha],A_predictions[trade],weight_A,A_open_price)
    print("-"*80)
    print("trade2")
    B_signal,B_predict = predicted_weighted_returns(B_x_val_multi[alpha],B_y_val_all[alpha],B_predictions[trade],weight_B,B_open_price)
    print("-"*80)
    print("trade3")
    C_signal,C_predict = predicted_weighted_returns(C_x_val_multi[alpha],C_y_val_all[alpha],C_predictions[trade],weight_C,C_open_price)
    plot_returns.append(total_returns)
    A_signals.append(A_signal)
    B_signals.append(B_signal)
    C_signals.append(C_signal)


    print("final returns is {}".format(total_returns))


plt.title("Weightings BT VOD TEF")
plt.plot(weights_array)
####NEW PLOTTING METHOD
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(plot_returns, 'b-')
ax2.plot(A_signals, 'r')
ax2.plot(B_signals, 'g')
ax2.plot(C_signals, 'y')
ax1.set_xlabel('Trades')
ax1.set_ylabel('% of start price - returns', color='b')
ax2.set_ylim([-7,7])
ax2.set_ylabel('position', color='r')
plt.title('portfolio returns (2% anti-symmetric threshold)')
plt.show()
