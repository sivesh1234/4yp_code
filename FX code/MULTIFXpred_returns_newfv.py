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
from numpy import asarray
from numpy import save
data = pd.read_csv("fxData.csv")
print(data.head())


target_pair = 'AUDUSD'
pair2 = "USDJPY"
pair3 = "EURUSD"



data[target_pair].plot()
#Moving average parameters
short_window = 41
long_window = 101
#Set anticausal shift
shift = 0
#Set trade horizon
trade_horizon = 5






#Create moving average signal
data['signal'] = 0.0
data['short_mavg'] = data[target_pair].rolling(window=short_window,
                                              min_periods=1,center=False).mean()


data['long_mavg'] = data[target_pair].rolling(window=long_window,min_periods=1,center=False).mean()
data['short_mavg'] = data['short_mavg'].shift(periods=(shift))
data['long_mavg'] = data['long_mavg'].shift(periods=(shift))





data['signal'][short_window:] = np.where(data['short_mavg'][short_window:] > data['long_mavg'][short_window:], 1.0, 0.0)

df = data

features_considered = ['signal',target_pair, pair2, pair3]
features = df[features_considered]
features.plot(subplots=True)
plt.show()
dataset = features.values

TRAIN_SPLIT = 20000
BATCH_SIZE = 100 #Batch size no. of periods fed intro training
BUFFER_SIZE = 10000 #Buffer size greater than sample size to ensure perfect shuffle
EVALUATION_INTERVAL = 200 #???
EPOCHS = 3


def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title("Multi-stock model training and validation loss")
  plt.legend()

  plt.show()


def multivariate_data(dataset_full,dataset, target,  start_index, end_index, history_size,
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
      full_y.append(dataset_full[i+target_size])
    else:
      full_y.append(dataset_full[i:i+target_size])
  return np.array(data), np.array(labels), np.array(full_y)
past_history = long_window

#periods fed into prediction for testing
#periods predicted for testing
STEP = 1 #How many periods it samples over
def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps
x_train_multi, y_train_multi, x_val_all = multivariate_data(dataset,dataset[:,[1,2,3]], dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 trade_horizon, STEP)
 #Returns testing data set = x and labels = y
x_val_multi, y_val_multi, y_val_all = multivariate_data(dataset,dataset[:,[1,2,3]], dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             trade_horizon, STEP)
final_returns = []
total_returns = 0
plot_returns = []
multiple_returns = []
all_signals = []
open_price = x_val_multi[0][0][0]
print("open price is {}".format(open_price))


def predicted_returns(history, true_future, prediction):
  #This randomly decides a buy,sell,hold

  signal = 0
  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  start = history[long_window-1][0]


  std = np.std(history[:][1])
  end = true_future[trade_horizon-1][1]

  start_pred = prediction[0]

  prediction_average = np.mean(prediction)
  predicted_end = prediction[trade_horizon-1]
  #This sets the signal to the best option
  print("predicted_average {}".format(prediction_average))
  if prediction_average > 0.5:
      signal = 1
      print("long")
  elif prediction_average < 0.05:
      signal = -1
      print("short")
  else:
      signal = 0
  returns = ((end - start)/open_price)*100
  returns = returns*signal
  global final_returns
  global total_returns
  global plot_returns
  global all_signals
  total_returns = total_returns + returns
  print(total_returns)
  # print(total_returns)
  plot_returns.append(total_returns)
  all_signals.append(signal)
  return signal, returns, prediction_average






def multi_step_plot(history, true_future, prediction,signal=0,returns=0,trade=0):


###Initialise main figure
  plt.figure(0)

##First plot shows moving average position prediction
  fig1 = plt.subplot(411)
  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  start = history[long_window-1][0]
  start_pred = prediction[0]
  difference = start - start_pred
  prediction = prediction + difference
  # plt.plot(num_in, np.array(history[:,0]), label='History')
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
  plt.plot(num_in, np.array(history[:,0]), label='{} History'.format(target_pair))
  plt.plot(np.arange(num_out)/STEP, np.array(true_future[:,1]), 'bo',
           label='True Future')

  plt.legend(loc='upper left')
  plt.subplot(413)
  plt.plot(num_in, np.array(history[:,1]), label='{} History'.format(pair2))
  plt.plot(np.arange(num_out)/STEP, np.array(true_future[:,2]), 'bo',
           label='True Future')

  plt.legend(loc='upper left')
  plt.subplot(414)
  plt.plot(num_in, np.array(history[:,2]), label='{} History'.format(pair3))
  plt.plot(np.arange(num_out)/STEP, np.array(true_future[:,3]), 'bo',
           label='True Future')

  plt.legend(loc='upper left')


  ###DISPLAYS PLOTS

  plt.show(block=False)
  plt.pause(0.5)
  plt.close()


  ###SAVES PLOTS TO FILE
  # file_name='BT-multi_figures/figure{}.png'.format(trade)
  # plt.savefig(file_name)
  # plt.clf()






# #Building the model
# model = tf.keras.models.Sequential()
#
# #Adding layers - two LSTM layers and a dense output layer
# model.add(tf.keras.layers.LSTM(64,return_sequences=True,input_shape=x_train_multi.shape[-2:]))
# #Return_sequences: whether to return the last output in output sequence or full sequence
#
# #Reshaping with negative integers works out the integer required
# #e.g. [3,4,5] reshaped with [-1,10] goes to [6,10]
# model.add(tf.keras.layers.LSTM(64, activation='relu'))
# model.add(tf.keras.layers.Dense(trade_horizon))  #Output has 10 as parameter as making 10 predictions
# #Compile model
# print(model.summary())
#
# #RMSprop optimizer divides the gradient by a running average of its recent magnitude
# #mae is mean absolute error
# model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

#Fitting/training the model
# multi_step_history = model.fit(train_data_multi, epochs=EPOCHS,
#                                          steps_per_epoch=EVALUATION_INTERVAL,
#                                          validation_data=val_data_multi,
#                                          validation_steps=50)



#New fitting step
# multi_step_history = model.fit(x_train_multi, y_train_multi,
#                     validation_data=(x_val_multi, y_val_multi),
#                     epochs=10, batch_size=100, verbose=1)


# plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
# model.save('saved_model/{}_multi1-0_model'.format(target_pair))
model = tf.keras.models.load_model('saved_model/{}_multi1-0_model'.format(target_pair))

prediction_array = []
for alpha in range(0,5000,trade_horizon):
    trade = alpha/trade_horizon
    print(alpha/trade_horizon)
    ###Need to create prediction here
    pred = model.predict(x_val_multi)[alpha]
    # pred = y_val_all[:][:][0]
    global total_returns
    global plot_returns
    global multiple_returns
    signal,returns, prediction_average = predicted_returns(x_val_multi[alpha],y_val_all[alpha],pred)
    multi_step_plot(x_val_multi[alpha],y_val_all[alpha],pred,signal,returns,trade)
    prediction_array.append(prediction_average)

save('{}_predictions.npy'.format(target_pair),prediction_array)

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(plot_returns, 'b-')
ax2.plot(all_signals, 'r')
ax1.set_xlabel('Trades')
ax1.set_ylabel('% of start price - returns', color='b')
ax2.set_ylim([-7,7])
ax2.set_ylabel('position', color='r')
plt.title('{} ({}/{}) MACD returns. Trade horizon {}'.format(target_pair,short_window,long_window,trade_horizon))
plt.show()
