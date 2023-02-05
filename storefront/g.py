import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas_datareader as pdr
from sklearn.neighbors import KNeighborsClassifier
from datetime import date
from dateutil.relativedelta import relativedelta
import mpld3
from joblib import dump
import os

def LSTM():
    data = Data
    new_data = pd.DataFrame(index=range(0,len(data)),columns=['Close'])
    for i in range(0,len(data)):
        new_data['Close'][i] = data.Close[i]
    dataset = new_data.values
    train = dataset[0:int(len(new_data)*0.9),:]
    valid = dataset[int(len(new_data)*0.9):,:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    model = Sequential()
    model.add(layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(layers.LSTM(units=50))
    model.add(layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=5, batch_size=60, verbose=0)
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)
    X_test = []
    Y_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
        Y_test.append(inputs[i,0])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing = scaler.inverse_transform(closing_price)
    fig = plt.figure(figsize=(10,6), dpi=100)
    plt.plot(scaler.inverse_transform(Y_test.reshape(-1,1)), color = 'red', label = 'Actual Stock Price')
    plt.plot(closing, color = 'green', label = 'Predicted Stock Price')
    plt.title(Stock+" Stock Price Prediction Using LSTM")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+Stock+'\\'+Stock+"lstm.txt"
    path = open(path, 'w')
    path.write(mpld3.fig_to_html(fig))
    path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+Stock+'\\'+Stock+"metrics.txt"
    path = open(path, 'a')
    path.write("\nLSTM")
    path.write("\n"+str(metrics.max_error(Y_test, closing_price))+"\n"+
               str(metrics.mean_absolute_error(Y_test, closing_price))+"\n"+
               str(metrics.mean_squared_error(Y_test, closing_price))+"\n"+
               str(metrics.mean_absolute_percentage_error(Y_test, closing_price))+"\n"+
               str(metrics.median_absolute_error(Y_test, closing_price))+"\n"+
               str(metrics.r2_score(Y_test, closing_price))+"\n")
    

def SVM():
    TrainData = Data.Close[0:int(len(Data)*0.90)-15]
    TrainDataPrediction = Data.Close[15:int(len(Data)*0.90)]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(Data.Close).reshape(-1,1))
    X = np.array(TrainData).reshape(-1,1)
    y = np.array(TrainDataPrediction)
    svr = SVR(kernel='rbf', C=1000, gamma=0.05)
    svr.fit(X,y)
    forecastData = np.array(Data.Close[int(len(Data)*0.90)-15:-15]).reshape(-1,1)
    tD = np.array(Data.Close[int(len(Data)*0.90):])
    sP = svr.predict(forecastData)
    trueData = scaler.transform(np.array(tD).reshape(-1,1))
    svm_prediction = scaler.transform(np.array(sP).reshape(-1,1))
    fig = plt.figure(figsize=(10,6), dpi=100)
    plt.plot(tD, color = 'red', label = 'Actual Stock Price')
    plt.plot(sP, color = 'green', label = 'Predicted Stock Price')
    plt.title(Stock+" Stock Price Prediction Using SVM")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+Stock+'\\'+Stock+"svm.txt"
    path = open(path, 'w')
    path.write(mpld3.fig_to_html(fig))
    path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+Stock+'\\'+Stock+"metrics.txt"
    path = open(path, 'a')
    path.write("\nSVM")
    path.write("\n"+str(metrics.max_error(trueData, svm_prediction))+"\n"+
               str(metrics.mean_absolute_error(trueData, svm_prediction))+"\n"+
               str(metrics.mean_squared_error(trueData, svm_prediction))+"\n"+
               str(metrics.mean_absolute_percentage_error(trueData, svm_prediction))+"\n"+
               str(metrics.median_absolute_error(trueData, svm_prediction))+"\n"+
               str(metrics.r2_score(trueData, svm_prediction))+"\n")
    
    
def linearReg():
    X = np.arange(0,len(Data)).reshape(-1,1)
    Y = Data.Close
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    scaler = StandardScaler().fit(X_train)
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    fig = plt.figure(figsize=(10,6), dpi=100)
    ax = plt.axes()
    ax.plot(np.arange(0,len(Data)), Data.Close, color = 'red', label='Actual Stock Price')
    ax.plot(X_train, lm.predict(X_train).T, color = 'green', label='Predicted Stock Price')
    ax.axis('tight')
    plt.title(Stock+" Stock Price Prediction Using Linear Regression")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+Stock+'\\'+Stock+"linearReg.txt"
    path = open(path, 'w')
    path.write(mpld3.fig_to_html(fig))
    path = 'D:\Prajwal\Python\Django\storefront\storefront\models\\'+Stock+'\\'+Stock+'linearReg.joblib'
    dump(lm, path)

def KNN():
    data = Data
    data["Diff"] = data.Close.diff()
    data["SMA_2"] = data.Close.rolling(2).mean()
    data["Force_Index"] = data["Close"] * data["Volume"]
    data["y"] = data["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
    data = data.drop(["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"], axis=1,).dropna()
    X = data.drop(["y"], axis=1).values
    y = data["y"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    fig = plt.figure(figsize=(10,6), dpi=100)
    plt.plot(y_test[int(len(y_test)/2):], color = 'red', label = 'Actual Stock Price')
    plt.plot(y_pred[int(len(y_test)/2):], color = 'green', label = 'Predicted Stock Price')
    plt.title(Stock+" Stock Price Prediction Using KNN")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+Stock+'\\'+Stock+"knn.txt"
    path = open(path, 'w')
    path.write(mpld3.fig_to_html(fig))
    path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+Stock+'\\'+Stock+"metrics.txt"
    path = open(path, 'a')
    path.write("\nKNN")
    path.write("\n"+str(metrics.accuracy_score(y_test, y_pred))+"\n"+
               str(metrics.f1_score(y_test, y_pred))+"\n"+
               str(metrics.recall_score(y_test, y_pred))+"\n"+
               str(metrics.precision_score(y_test, y_pred))+"\n")

def LSTMPredict():
    new_data = pd.DataFrame(index=range(0,len(Data)),columns=['Close'])
    for i in range(0,len(Data)):
        new_data['Close'][i] = Data.Close[i]
    dataset = new_data.values
    train = dataset[0:,:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    model = Sequential()
    model.add(layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(layers.LSTM(units=50))
    model.add(layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=5, batch_size=60, verbose=0)
    path = 'D:\Prajwal\Python\Django\storefront\storefront\models\\'+Stock+'\\'+Stock+'lstm.joblib'
    dump(model, path)
    fig = plt.figure(figsize=(10,6), dpi=100)
    plt.plot(np.array(Data.Close).reshape(-1,1), color = 'red', label = 'Actual Stock Price')
    plt.title(Stock+" Stock Price")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+Stock+'\\'+Stock+".txt"
    path = open(path, 'w')
    path.write(mpld3.fig_to_html(fig))

def SVMPredict():
    TrainData = Data.Close[0:-15]
    TrainDataPrediction = Data.Close[15:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(Data.Close).reshape(-1,1))
    X = np.array(TrainData).reshape(-1,1)
    y = np.array(TrainDataPrediction)
    svr = SVR(kernel='rbf', C=1000, gamma=0.05)
    svr.fit(X,y)
    path = 'D:\Prajwal\Python\Django\storefront\storefront\models\\'+Stock+'\\'+Stock+'svm.joblib'
    dump(svr, path)

def KNNPredict():
    data = Data
    data["Diff"] = data.Close.diff()
    data["SMA_2"] = data.Close.rolling(2).mean()
    data["Force_Index"] = data["Close"] * data["Volume"]
    data["y"] = data["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
    data = data.drop(["Open", "High", "Low", "Close", "Volume", "Diff", "Dividends", "Stock Splits"], axis=1,).dropna()
    X = data.drop(["y"], axis=1).values
    y = data["y"].values
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    path = 'D:\Prajwal\Python\Django\storefront\storefront\models\\'+Stock+'\\'+Stock+'knn.joblib'
    dump(knn, path)
    

def initiate(Stk):
    os.mkdir('D:\Prajwal\Python\Django\storefront\storefront\data\\'+Stk)
    os.mkdir('D:\Prajwal\Python\Django\storefront\storefront\models\\'+Stk)
    global Stock, Data
    Stock = Stk
    Check = yf.Ticker(Stock)
    Data = Check.history(start = date.today() - relativedelta(years=int(10)), end = date.today())[:-1]
    LSTM()
    SVM()
    linearReg()
    KNN()
    LSTMPredict()
    SVMPredict()
    KNNPredict()
