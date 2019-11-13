#Built using https://mapr.com/blog/deep-learning-tensorflow/
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import shutil
from pandas_datareader import data as pdr
import datetime



aapl = pdr.get_data_yahoo('NGG',
                          start=datetime.datetime(2007, 10, 26),
                          end=datetime.datetime(2019, 10, 26))
ts = aapl['Close']
# ts.plot(title='National grid share price')
# plt.show()

TS = np.array(ts)
num_periods = 20
f_horizon = 1
x_data = TS[:(len(TS)-(len(TS)%num_periods))]
x_batches = x_data.reshape(-1,20,1)
y_data = TS[1:(len(TS)-(len(TS)%num_periods))+f_horizon]
y_batches = y_data.reshape(-1,20,1)


def test_data(series,forecast,num_periods):
    test_x_setup = TS[-(num_periods+forecast):]
    testX = test_x_setup[:num_periods].reshape(-1,20,1)
    testY = TS[-(num_periods):].reshape(-1,20,1)
    return testX,testY

X_test, Y_test = test_data(TS,f_horizon,num_periods)
# tf.reset_default_graph()
num_periods = 20
inputs = 1
hidden = 100
output = 1

# X = tf.compat.v1.placeholder(tf.float32, [None,num_periods,inputs])
# Y = tf.compat.v1.placeholder(tf.float32, [None,num_periods,inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden,activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
learning_rate=0.001

stacked_rnn_output = tf.reshape(rnn_output, [-1,hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output,output)
outputs = tf.reshape(stacked_outputs, [-1,num_periods,output])

loss = tf.reduce_sum(tf.square(outputs -y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initalizer()


epochs = 10000

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE:",mse)
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    print(y_pred)





#plot
plt.title("Forecast vs Actual")
plt.plot(pd.Series(np.ravel(Y_test)),"bo",label="Actual")
plt.plot(pd.Series(np.ravel(y_pred)),"r.",label="Forecast")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")
