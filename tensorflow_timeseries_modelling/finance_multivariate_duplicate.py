#Weather forecast timeseries RNN MULTIVARIATE


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

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
                          start=datetime.datetime(2010, 10, 26),
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
#unsure what short_window does here***************
vod['signal'][short_window:] = np.where(vod['short_mavg'][short_window:] > vod['long_mavg'][short_window:], 1.0, 0.0)







df = vod

features_considered = ['signal','Close']
features = df[features_considered]
features.index = df.index
features.plot(subplots=True)
dataset = features.values
TRAIN_SPLIT = 1500
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std


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


BATCH_SIZE = 256
BUFFER_SIZE = 10000
EVALUATION_INTERVAL = 200
EPOCHS = 1

#Samples every 6 periods, label for a datapoint is 72 periods into the future
#Network shown 720 periods

past_history = 50 #periods fed into prediction
future_target = 10 #periods predicted
STEP = 1 #How many periods it samples over
#SINGLE STEP Model


# x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
#                                                    TRAIN_SPLIT, past_history,
#                                                    future_target, STEP,
#                                                    single_step=True)
# x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
#                                                TRAIN_SPLIT, None, past_history,
#                                                future_target, STEP,
#                                                single_step=True)
#
#
#
# train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
# train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
#
# val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
# val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
#
# single_step_model = tf.keras.models.Sequential()
# single_step_model.add(tf.keras.layers.LSTM(32,
#                                            input_shape=x_train_single.shape[-2:]))
# single_step_model.add(tf.keras.layers.Dense(1))
#
# single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
# single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
#                                             steps_per_epoch=EVALUATION_INTERVAL,
#                                             validation_data=val_data_single,
#                                             validation_steps=50)
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
# def show_plot(plot_data, delta, title):
#   labels = ['History', 'True Future', 'Model Prediction']
#   marker = ['.-', 'rx', 'go']
#   time_steps = create_time_steps(plot_data[0].shape[0])
#   if delta:
#     future = delta
#   else:
#     future = 0
#
#   plt.title(title)
#   for i, x in enumerate(plot_data):
#     if i:
#       plt.plot(future, plot_data[i], marker[i], markersize=10,
#                label=labels[i])
#     else:
#       plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
#   plt.legend()
#   plt.xlim([time_steps[0], (future+5)*2])
#   plt.xlabel('Time-Step')
#   return plt
# for x, y in val_data_single.take(3):
#   plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
#                     single_step_model.predict(x)[0]], 12,
#                    'Single Step Prediction')
#   plot.show()










 #Multistep model

#Creates training data. end index is trainsplit
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
#Creates validation data. start index is train split
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)



#????????
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history)) #creates an array of -history -> 0
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()




for x, y in train_data_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]))





multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
#Reshaping with negative integers works out the integer required
#e.g. [3,4,5] reshaped with [-1,10] goes to [6,10]


multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(10))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')


for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
