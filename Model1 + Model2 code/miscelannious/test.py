
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


dataset = [[1,2,3],[4,5,6],[7,8,9]]
dataset = np.array(dataset)
print(dataset[1][0])


data = [10,10,10,10,20,20,20,20,20]
print(data[4:])
