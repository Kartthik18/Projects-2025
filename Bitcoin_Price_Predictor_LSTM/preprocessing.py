import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(path="data/btc.csv"):
    df = pd.read_csv(path)
    closedf = df[['Date', 'Close']]
    closedf = closedf.drop('Date', axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))
    return closedf, scaler

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
