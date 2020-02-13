#Finance forecast timeseries RNN



from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import datetime
from pandas_datareader import data as pdr
#Matplot lib stuff
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
#Getting data
df = pdr.get_data_yahoo('AAPL',
                          start=datetime.datetime(2000, 10, 26),
                          end=datetime.datetime(2019, 10, 26))


#returns windows of time for training, history = past size of window,
#Target = how far in future for predicition, = label that needs to be predicted
def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)
# How many rows are for training
TRAIN_SPLIT = 3000


uni_data = df['Close']
uni_data.index = df.index

# uni_data.head()
# uni_data.plot(subplots=True)
# plt.show()
uni_data = uni_data.values

#Need to normalise features before training a neural network
# uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
# uni_train_std = uni_data[:TRAIN_SPLIT].std()
# uni_data = (uni_data-uni_train_mean)/uni_train_std


#Give 20 last temps to predict next time step

univariate_past_history = 20
univariate_future_target = 20




x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)
# print ('Single window of past history')
# print (x_train_uni[0])
# print ('\n Target temperature to predict')
# print (y_train_uni[0])



#Set up a plot for past 20 and then true value
def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps
def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

# show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')




#Set up base average to compare prediction to
def baseline(history):
  return np.mean(history)
#
# show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
#            'Baseline Prediction Example')
# plt.show()





#Building RNN using tf.data to shuffle, batch and cache all the data

BATCH_SIZE = 128
BUFFER_SIZE = 200

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')
for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)
EVALUATION_INTERVAL = 1500
EPOCHS = 1

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
for x, y in val_univariate.take(10):
  plot = show_plot([x[10].numpy(), y[10].numpy(),
                    simple_lstm_model.predict(x)[20]], 0, 'Simple LSTM model')
  plot.show()
