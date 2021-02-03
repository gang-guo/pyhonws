from csv import Dialect

import numpy as np
import json

import pandas

import csv

Dialect





def loaddata(datafile, len, sep=' '):
   data = np.fromfile(datafile, sep=sep)
   data = data.reshape([data.shape[0] // len, len])
   ratio = 0.8
   offset = int(data.shape[0] * ratio)
   train_data = data[:offset]
   maximum = train_data.max(axis=0)
   minimum = train_data.min(axis=0)
   for i in range(len):
      data[:,i] = (data[:,i] - minimum[i]) / (maximum[i] - minimum[i])
   train_data = data[:offset]
   test_data = data[offset:]
   return train_data,test_data,maximum,minimum