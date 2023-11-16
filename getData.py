#code for getting data from csv; 
import pandas as pd
import datetime as dt
import os
import numpy as np
datafile = './Data/data.csv' #replace data with name of file
if not os.path.exists(datafile):
    exit
stockprices = pd.read_csv(datafile, parse_dates=['Date'])
#sort by date
stockprices = stockprices.sort_values('Date')
print(stockprices.head(5))


#create training and test data
test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))
print(f"train_size: {train_size}")
print(f"test_size: {test_size}")

train = stockprices[:train_size][["Close"]]
test = stockprices[train_size:][["Close"]]