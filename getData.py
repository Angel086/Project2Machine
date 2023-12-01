#code for getting data from csv; 
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import numpy as np

def read_data():
    datafile = './Data/HistoricalData_1699922246348.csv' #replace data with name of file
    if not os.path.exists(datafile):
        exit
    stockprices = pd.read_csv(datafile, parse_dates=['Date']) #parse_dates so it sorts by date correctly
    #sort by date
    stockprices = stockprices.sort_values('Date')

    #set ratio of test to training data points
    test_ratio = 0.2
    training_ratio = 1 - test_ratio

    train_size = int(training_ratio * len(stockprices))
    test_size = int(test_ratio * len(stockprices))
    print(f"train_size: {train_size}")
    print(f"test_size: {test_size}")

    cols = ['Close/Last']
    if(set(stockprices.columns).issuperset(cols)):
        print("true")

    #create test and training data points based of closing price. 
    test = stockprices[train_size:][["Close/Last"]]
    train = stockprices[:train_size][["Close/Last"]]
    return test,train