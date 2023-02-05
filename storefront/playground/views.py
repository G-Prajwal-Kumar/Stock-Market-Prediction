from django.shortcuts import render
from django.http import HttpResponse
import os
from joblib import load
import yfinance as yf
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from g import *

# Create your views here.

def home(request):
    return render(request, 'index.html')

def image(request, stockName, model):
    path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+stockName+'\\'+stockName+model+'.txt'
    if(stockName == model):
        path = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+stockName+'\\'+stockName+'.txt'
    temp = open(path, 'r')
    return HttpResponse(temp.read())

def train(request):
    stockName = request.GET['stockName']
    modelPath = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+stockName
    Data = yf.Ticker(stockName).history(start = date.today() - relativedelta(years=int(10)), end = date.today())
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(Data.Close).reshape(-1,1))
    if(os.path.exists(modelPath)):
        dataPath = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+stockName+'\\'+stockName+'metrics.txt'
        data = open(dataPath, 'r')
        data = data.readlines()
        lstm = data[2:8]
        svm = data[10:16]
        knn = data[18:]
        lstmmodel = load('D:\Prajwal\Python\Django\storefront\storefront\models\\'+stockName+'\\'+stockName+'lstm.joblib')
        svmmodel = load('D:\Prajwal\Python\Django\storefront\storefront\models\\'+stockName+'\\'+stockName+'svm.joblib')
        knnmodel = load('D:\Prajwal\Python\Django\storefront\storefront\models\\'+stockName+'\\'+stockName+'knn.joblib')
        lrmodel = load('D:\Prajwal\Python\Django\storefront\storefront\models\\'+stockName+'\\'+stockName+'linearReg.joblib')
        lstmd = np.array(scaled_data[-60:]).reshape(-1,1)
        lstmd = np.reshape(lstmd, (lstmd.shape[1],lstmd.shape[0],1))
        svmd = np.array(Data.Close[-15].reshape(-1,1))
        knndata = yf.Ticker(stockName).history(start = date.today() - relativedelta(years=int(1)), end = date.today())
        knndata["Diff"] = knndata.Close.diff()
        knndata["SMA_2"] = knndata.Close.rolling(2).mean()
        knndata["Force_Index"] = knndata["Close"] * knndata["Volume"]
        knndata["y"] = knndata["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
        knndata = knndata.drop(["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Diff"], axis=1,).dropna()
        knnd = knndata.drop(["y"], axis=1).values
        lstmout = scaler.inverse_transform(lstmmodel.predict(lstmd))
        svmout = svmmodel.predict(svmd)
        knnout = knnmodel.predict(knnd)
        lrout = lrmodel.predict(np.array(len(Data)).reshape(-1,1))
        return render(request, 'train.html', {'stockName': stockName,
            'lstmme': lstm[0],
            'lstmmae': lstm[1],
            'lstmmse': lstm[2],
            'lstmmape': lstm[3],
            'lstmMae': lstm[4],
            'lstmrs': lstm[5],
            'svmme': svm[0],
            'svmmae': svm[1],
            'svmmse': svm[2],
            'svmmape': svm[3],
            'svmMae': svm[4],
            'svmrs': svm[5],
            'accuracy': knn[0],
            'f1': knn[1],
            'rs': knn[2],
            'ps': knn[3],
            'LSTM': lstmout[0][0],
            'SVM': svmout[0],
            'KNN': knnout[-1],
            'linearReg': lrout[0]
        })
    else:
        initiate(stockName)
        dataPath = 'D:\Prajwal\Python\Django\storefront\storefront\data\\'+stockName+'\\'+stockName+'metrics.txt'
        data = open(dataPath, 'r')
        data = data.readlines()
        lstm = data[2:8]
        svm = data[10:16]
        knn = data[18:]
        lstmmodel = load('D:\Prajwal\Python\Django\storefront\storefront\models\\'+stockName+'\\'+stockName+'lstm.joblib')
        svmmodel = load('D:\Prajwal\Python\Django\storefront\storefront\models\\'+stockName+'\\'+stockName+'svm.joblib')
        knnmodel = load('D:\Prajwal\Python\Django\storefront\storefront\models\\'+stockName+'\\'+stockName+'knn.joblib')
        lrmodel = load('D:\Prajwal\Python\Django\storefront\storefront\models\\'+stockName+'\\'+stockName+'linearReg.joblib')
        lstmd = np.array(scaled_data[-60:]).reshape(-1,1)
        lstmd = np.reshape(lstmd, (lstmd.shape[1],lstmd.shape[0],1))
        svmd = np.array(Data.Close[-15].reshape(-1,1))
        knndata = yf.Ticker(stockName).history(start = date.today() - relativedelta(years=int(1)), end = date.today())
        knndata["Diff"] = knndata.Close.diff()
        knndata["SMA_2"] = knndata.Close.rolling(2).mean()
        knndata["Force_Index"] = knndata["Close"] * knndata["Volume"]
        knndata["y"] = knndata["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
        knndata = knndata.drop(["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Diff"], axis=1,).dropna()
        knnd = knndata.drop(["y"], axis=1).values
        lstmout = scaler.inverse_transform(lstmmodel.predict(lstmd))
        svmout = svmmodel.predict(svmd)
        knnout = knnmodel.predict(knnd)
        lrout = lrmodel.predict(np.array(len(Data)).reshape(-1,1))
        return render(request, 'train.html', {'stockName': stockName,
            'lstmme': lstm[0],
            'lstmmae': lstm[1],
            'lstmmse': lstm[2],
            'lstmmape': lstm[3],
            'lstmMae': lstm[4],
            'lstmrs': lstm[5],
            'svmme': svm[0],
            'svmmae': svm[1],
            'svmmse': svm[2],
            'svmmape': svm[3],
            'svmMae': svm[4],
            'svmrs': svm[5],
            'accuracy': knn[0],
            'f1': knn[1],
            'rs': knn[2],
            'ps': knn[3],
            'LSTM': lstmout[0][0],
            'SVM': svmout[0],
            'KNN': knnout[-1],
            'linearReg': lrout[0]
        })
        