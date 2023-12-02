import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


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