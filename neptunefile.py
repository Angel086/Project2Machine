#code for getting data from csv; 
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import neptune as nep
from neptune.integrations.tensorflow_keras import NeptuneCallback as ncb
from tensorflow.keras import Input
from keras.layers import LSTM, Dense
from keras import Model

window_size = 50

##Neptune run instances
run_sma = nep.init_run(
        project="nobodyofinterest/StockPricePrediction",
        name="SMA",
        description="Stock-prediction-machine-learning",
    tags=["stockprediction", "MA_Simple", "neptune"],
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MmNjYTY1ZS0wYmJhLTQ1MWQtOTYxMy02M2NiNjM2OTI5NGIifQ==",
    )

run_ema = nep.init_run(
        project="nobodyofinterest/StockPricePrediction",
        name="EMA",
        description="Stock-prediction-machine-learning",
    tags=["stockprediction", "MA_Simple", "neptune"],
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MmNjYTY1ZS0wYmJhLTQ1MWQtOTYxMy02M2NiNjM2OTI5NGIifQ==",
    )

run_lstm = nep.init_run(
        project="nobodyofinterest/StockPricePrediction",
        name="lstm",
        description="Stock-prediction-machine-learning",
    tags=["stockprediction", "MA_Simple", "neptune"],
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MmNjYTY1ZS0wYmJhLTQ1MWQtOTYxMy02M2NiNjM2OTI5NGIifQ==",
    )

datafile = './Data/data.csv' #replace data with name of file
if not os.path.exists(datafile):
    exit
stockprices = pd.read_csv(datafile, parse_dates=['Date']) #parse_dates so it sorts by date correctly

stockprices['Close/Last'] = stockprices['Close/Last'].replace('[\$,]', '', regex=True).astype(float)
stockprices = stockprices[::-1].reset_index(drop=True)

#sort by date
stockprices = stockprices.sort_values('Date')
print(stockprices.head(10))

def extract_seqZ_outComeY(data, N, offset):
    X, y = [], []
    for i in range (offset,len(data)):
        X.append(data[i-N: i])
        y.append(data[i])

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
    

def calculate_perf_metrics(var, train_size, run):
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

def plot_stock_trend(var, cur_title, stockprices, run):
    ax = stockprices[["Close/Last", var, "200 Days", "1 Year"]].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis("tight")
    plt.ylabel("Stock Price ($)")

    ## Log to Neptune
    run["Plot of Stock Predictions"].upload(
        nep.types.File.as_image(ax.get_figure())
    )
    
def run_LSTM(train, layer_units = window_size):
    input_layer = Input(shape=(train.shape[1],1))
    hidden_layer =LSTM(units=layer_units,return_sequences=True)(input_layer)
    hidden_layer =LSTM(units=layer_units)(hidden_layer)
    output_layer = Dense(1, activation="linear")(hidden_layer)
    model = Model(input_layer, output_layer)
    
    model.compile(loss = "mean_squared_error", optimizer = "adam")
    
    return model

def preprocess_test_data(data, scaler, window_size, test):
    raw = data["Close/Last"][len(data)-len(test)-window_size:].values
    raw = raw.reshape(-1,1)
    raw = scaler.transform(raw)
    
    test = [raw[i-window_size:i, 0]for i in range(window_size, raw.shape[0])]
    test = np.array(test)
    return test

def plot_LSTM_trend(train, test, run):
    figure = plt.figure((20,10))
    plt.plot(np.asarray(train.index), np.asarray(train["Close/Last"]), "Training Closing Price")
    plt.plot(np.asarray(test.index), np.asarray(test["Close/Last"]), "Test Closing Price")
    plt.plot(np.asarray(test.index), np.asarray(test["Predictions_LSTM"]), "Predicted Closing Price")
    plt.title("LTSM Model")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.legend(loc="upper left")
    
    run["Plot of Stock Predictions"].upload(neptune.types.File.as_image(figure))


##############################
#    End helper functions    #
##############################

def main():    
    
    
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

    run = nep.init_run(
        project="nobodyofinterest/StockPricePrediction",
        name="SMA",
        description="Stock-prediction-machine-learning",
    tags=["stockprediction", "MA_Simple", "neptune"],
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MmNjYTY1ZS0wYmJhLTQ1MWQtOTYxMy02M2NiNjM2OTI5NGIifQ==",
    )  # your credentials
    window_var = f"{window_size} Days"
    stockprices[window_var] = stockprices["Close/Last"].rolling(window_size).mean()

    stockprices["200 Days"] = stockprices["Close/Last"].rolling(200).mean()
    start_2023 = dt.datetime(2023, 1, 1)
    end_2023 = dt.datetime(2023, 12, 31)

    stockprices["1 Year"] = stockprices["Close/Last"].rolling(365).mean()
    plot_stock_trend(var=window_var, cur_title="Simple Moving Averages", stockprices=stockprices, run=run_sma)
    rmse_sma, mape_sma = calculate_perf_metrics(var=window_var, train_size=train_size, run=run_sma)

    ### Stop the run
    run.stop()



    # Initialize a Neptune run
    run = nep.init_run(
    project="nobodyofinterest/StockPricePrediction",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MmNjYTY1ZS0wYmJhLTQ1MWQtOTYxMy02M2NiNjM2OTI5NGIifQ==",
        name="EMA",
        description="stock-prediction-machine-learning",
        tags=["stockprediction", "MA_Exponential", "neptune"],
    )

    ###### Exponential MA
    window_ema_var = f"{window_var}"

    # Calculate the 50-day exponentially weighted moving average
    stockprices[window_ema_var] = (
        stockprices["Close/Last"].ewm(span=window_size, adjust=False).mean()
    )
    stockprices["200 Days"] = stockprices["Close/Last"].rolling(200).mean()

    stockprices["1 Year"] = stockprices["Close/Last"].rolling(365).mean()

    ### Plot and performance metrics for EMA model
    plot_stock_trend(var=window_ema_var, cur_title="Exponential Moving Averages", stockprices=stockprices, run=run_ema)
    rmse_ema, mape_ema = calculate_perf_metrics(var=window_ema_var, train_size=train_size, run=run_ema)

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
    run = nep.init_run(
        project="nobodyofinterest/StockPricePrediction",
        name="LSTM",
        description="stock-prediction-machine-learning",
        tags=["stockprediction", "LSTM", "neptune"],
    )
    run["LSTM_args"] = cur_LSTM_args

    scaled_training_data = scaled_data[: train.shape[0]]
    x_train, y_train =extract_seqZ_outComeY(scaled_training_data, window_size, window_size)

    neptune_callback = ncb(run=run)

    model = run_LSTM(x_train, layer_units)

    training_history = model.fit(
        x_train,
        y_train,
        cur_epochs,
        cur_batch_size,
        verbose=1,
        validation_split=0.1,
        shuffle=True,
        callbacks=[neptune_callback],
    )

    x_test = preprocess_test_data(stockprices, scaler, window_size, test)

    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    test["Predictions_LSTM"] = predicted_price

    LSTM_rmse, LSTM_mape = calculate_perf_metrics(np.array(test["Close/Last"]), np.array(test["Predictions_lstm"]), train_size, run_lstm)

    run["RMSE"] = LSTM_rmse
    run["MAPE (%)"] = LSTM_mape
    
    plot_LSTM_trend(train, test, run_lstm)
    
    run.stop()
    
if __name__ == "__main__":
    main();