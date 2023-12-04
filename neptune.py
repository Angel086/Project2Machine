#code for getting data from csv; 
################
##
#Rename file to somethig else, due to needing the neptune library, can cause issues if not renamed
##
################

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import neptune





datafile = './Data/data.csv' #replace data with name of file
if not os.path.exists(datafile):
    exit
stockprices = pd.read_csv(datafile, parse_dates=['Date']) #parse_dates so it sorts by date correctly
stockprices['Close/Last'] = stockprices['Close/Last'].replace('[\$,]', '', regex=True).astype(float)

#sort by date
stockprices = stockprices.sort_values('Date')
print(stockprices.head(10))
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stockprices[["Close/Last"]])  # Assuming "Close/Last" is your column name


#set ratio of test to training data points
test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))
print(f"train_size: {train_size}")
print(f"test_size: {test_size}")

#create test and training data points based of closing price. 
train = stockprices[:train_size][["Close/Last"]]
test = stockprices[train_size:][["Close/Last"]]
#gather.extract_seqZ_outComeY(scaled_data, 10, 2)






def extract_seqZ_outComeY(data, N, offset):
    X, y = [], []
    for i in range (offset,len(data)):
        X.append(data[i-N: i])
        y.append(data[i])
    print(X[:5], y[:5])
    print([len(seq) for seq in X])  # Print the length of each sequence in X

    return np.array(X), np.array(y)

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape
    

def calculate_perf_metrics(var):
    ### RMSE
    rmse = calculate_rmse(
        np.array(stockprices[train_size:]["Close/Last"]),
        np.array(stockprices[train_size:][var]),
    )
    ### MAPE
    mape = calculate_mape(
        np.array(stockprices[train_size:]["Close/Last"]),
        np.array(stockprices[train_size:][var]),
    )

    ## Log to Neptune
    run["RMSE"] = rmse
    run["MAPE (%)"] = mape

    return rmse, mape

def plot_stock_trend(var, cur_title, stockprices=stockprices):
    ax = stockprices[["Close/Last", var, "200day"]].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis("tight")
    plt.ylabel("Stock Price ($)")

    ## Log to Neptune
    run["Plot of Stock Predictions"].upload(
        neptune.types.File.as_image(ax.get_figure())
    )
    
    
    
    
    
    
    
    
window_size = 50
run = neptune.init_run(
    project="aguerra688/StockPricePrediction",
    name="SMA",
    description="Stock-prediction-machine-learning",
   tags=["stockprediction", "MA_Simple", "neptune"],
   api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzcxZmUwMy04NDZmLTRmYWMtYTMwNy1mNDVhNjA0MzcyNjkifQ==",
)  # your credentials
window_var = f"{window_size}day"
stockprices[window_var] = stockprices["Close/Last"].rolling(window_size).mean()

stockprices["200day"] = stockprices["Close/Last"].rolling(200).mean()

plot_stock_trend(var=window_var, cur_title="Simple Moving Averages")
rmse_sma, mape_sma = calculate_perf_metrics(var=window_var)

### Stop the run
run.stop()



# Initialize a Neptune run
run = neptune.init_run(
   project="aguerra688/StockPricePrediction",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzcxZmUwMy04NDZmLTRmYWMtYTMwNy1mNDVhNjA0MzcyNjkifQ==",
    name="EMA",
    description="stock-prediction-machine-learning",
    tags=["stockprediction", "MA_Exponential", "neptune"],
)

###### Exponential MA
window_ema_var = f"{window_var}_EMA"

# Calculate the 50-day exponentially weighted moving average
stockprices[window_ema_var] = (
    stockprices["Close/Last"].ewm(span=window_size, adjust=False).mean()
)
stockprices["200day"] = stockprices["Close/Last"].rolling(200).mean()

### Plot and performance metrics for EMA model
plot_stock_trend(
    var=window_ema_var, cur_title="Exponential Moving Averages")
rmse_ema, mape_ema = calculate_perf_metrics(var=window_ema_var)

### Stop the run
run.stop()

layer_units = 50
optimizer = "adam"
cur_epochs = 15
cur_batch_size = 20

cur_LSTM_args = {
    "units": layer_units,
    "optimizer": optimizer,
    "batch_size": cur_batch_size,
    "epochs": cur_epochs,
}

# Initialize a Neptune run
run = neptune.init_run(
    project=myProject,
    name="LSTM",
    description="stock-prediction-machine-learning",
    tags=["stockprediction", "LSTM", "neptune"],
)
run["LSTM_args"] = cur_LSTM_args
