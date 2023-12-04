from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler


def main():
    datafile = './Data/HistoricalData_1699922246348.csv' #replace data with name of file
    if not os.path.exists(datafile):
        exit
    df = pd.read_csv(datafile, parse_dates=['Date']) #parse_dates so it sorts by date correctly
    #sort by date
    df = df.sort_values('Date')
    cols = ['Close/Last']
    set(df.columns).issuperset(cols)
    df[df.columns[1:]] = df[df.columns[1:]].replace('[\$,]', '', regex=True).astype(float)
    df.head()
    #separate into test and train data
    train_data = df[:2265]
    test_data = df[2265:]
    #scale data
    scaler = MinMaxScaler()
    train_data = train_data.values.reshape(-1,1)
    test_data = test_data.values.reshape(-1,1)
    # Train the Scaler with training data and smooth data
    smoothing_window_size = 600
    for di in range(0,2400,smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

    # You normalize the last bit of remaining data
    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
    
    # Reshape both train and test data
    train_data = train_data.values.reshape(-1)

    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)
    # Now perform exponential moving average smoothing
    # So the data will have a smoother curve than the original ragged data
    EMA = 0.0
    gamma = 0.1
    for ti in range(2265):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA

    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data,test_data],axis=0)

    window_size = 100
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size,N):

        if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx,'Date']

        std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
        std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))



if __name__ == "__main__":
    main()