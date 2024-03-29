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

stock1 = 'RBS.L'
stock2 = 'BARC.L'
stock3 = 'LLOY.L'
tar_data = pdr.get_data_yahoo(stock1,
                          start=datetime.datetime(1990, 10, 26),
                          end=datetime.datetime(2019, 10, 26)) #year, month, date

B_data = pdr.get_data_yahoo(stock2,
                          start=datetime.datetime(1990, 10, 26),
                          end=datetime.datetime(2019, 10, 26)) #year, month, date
C_data = pdr.get_data_yahoo(stock3,
                          start=datetime.datetime(1990, 10, 26),
                          end=datetime.datetime(2019, 10, 26))


short_window = 41
long_window = 101

#NEED TO MAKE ALL DATAFRAMES OF SAME LENGTH
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

tar_data['B_Close'] = B_data['Close']
tar_data['C_Close'] = C_data['Close']
tar_data['signal'] = 0.0
tar_data['short_mavg'] = tar_data['Close'].rolling(window=short_window,min_periods=1,center=False).mean()
tar_data['long_mavg'] = tar_data['Close'].rolling(window=long_window,min_periods=1,center=False).mean()
tar_data['short_mavg'] = tar_data['short_mavg'].shift(periods=(-50))
tar_data['long_mavg'] = tar_data['long_mavg'].shift(periods=(-50))

#if short > long then 'signal' = 1

tar_data['signal'][short_window:] = np.where(tar_data['short_mavg'][short_window:] > tar_data['long_mavg'][short_window:], 1.0, 0.0)

print("LENGTH OF DATA SET IS {}".format(len(tar_data['B_Close'])))


TRAIN_SPLIT = 3000


df = tar_data

features_considered = ['signal','Close','B_Close','C_Close']
features = df[features_considered]
features.index = df.index
features.plot(subplots=True)
plt.show()
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
  plt.title("Multi-stock model training and validation loss")
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
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
 #Returns testing data set = x and labels = y
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

train_data_multi = (x_train_multi, y_train_multi)
val_data_multi = (x_val_multi, y_val_multi)



#Defines plotting for multistep prediction
def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  start = history[89][1]
  start_10 = history[89][0]
  start_pred = prediction[0]
  difference = start_10 - start_pred
  # prediction = prediction + difference
  plt.plot(num_in, np.array(history[:, 0]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()


#Plots an example with no prediction
# for x, y in train_data_multi.take(1):
#   multi_step_plot(x[0], y[0], np.array([0]))


#Building the model
model = tf.keras.models.Sequential()

#Adding layers - two LSTM layers and a dense output layer
model.add(tf.keras.layers.LSTM(64,return_sequences=True,input_shape=x_train_multi.shape[-2:]))
#Return_sequences: whether to return the last output in output sequence or full sequence

#Reshaping with negative integers works out the integer required
#e.g. [3,4,5] reshaped with [-1,10] goes to [6,10]
model.add(tf.keras.layers.LSTM(64, activation='relu'))
model.add(tf.keras.layers.Dense(30))  #Output has 10 as parameter as making 10 predictions
#Compile model
print(model.summary())

#RMSprop optimizer divides the gradient by a running average of its recent magnitude
#mae is mean absolute error
model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

#Fitting/training the model
# multi_step_history = model.fit(train_data_multi, epochs=EPOCHS,
#                                          steps_per_epoch=EVALUATION_INTERVAL,
#                                          validation_data=val_data_multi,
#                                          validation_steps=50)



#New fitting step
multi_step_history = model.fit(x_train_multi, y_train_multi,
                    validation_data=(x_val_multi, y_val_multi),
                    epochs=10, batch_size=100, verbose=1)

plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')


#Save model

model.save('saved_model/{}_multi1-0_model'.format(stock1))

for alpha in range(0,1000,250):
    pred = model.predict(x_val_multi)[alpha]
    multi_step_plot(x_val_multi[alpha],y_val_multi[alpha],pred)
