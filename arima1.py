# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:02:53 2021

@author: Hangdong AN
## Data processing and model building for ARIMA models as well as data testing and visualisation of results.

"""
from pandas import DataFrame
import matplotlib.pyplot as plt
from pandas import read_csv
from datetime import datetime
from dateutil import tz  
import csv
import time
import datetime
from sklearn.metrics import accuracy_score
import os
from dateutil import tz
import pandas            as pd
import matplotlib.pyplot as plt
import tensorflow        as tf 
import numpy             as np
from scipy.signal          import medfilt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import matplotlib.pylab as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import Normalizer
from sklearn                import metrics

data = pd.read_csv('33.csv') # Data selection
ID2=data['SerialNumber'].unique()
data=data[['SystemSnapshotDate_UTC','SerialNumber','acchange']]
data=data[data['SerialNumber'] == ID2[15]]#Family data selection
data=data[['SystemSnapshotDate_UTC','acchange']]

dataset_new = data
for index, row in data.iterrows():
    dataset_new["SystemSnapshotDate_UTC"][index] = row["SystemSnapshotDate_UTC"].split(" ")[0]#Data day consumption aggregation
#print(dataset_new.head())
dataset_new_2 = dataset_new.groupby(by='SystemSnapshotDate_UTC')['acchange'].sum()*1

dict_dataset = {'SystemSnapshotDate_UTC':dataset_new_2.index,'acchange':dataset_new_2.values}#Reconfiguration of data
dataset_new_3 = pd.DataFrame(dict_dataset)
#print(dataset_new_3.head())
# dataset_new_3 = dataset_new_3.drop(dataset_new_3[dataset_new_3['acchange']<0.01].index)
dataset = dataset_new_3
dataset.set_index('SystemSnapshotDate_UTC',inplace=True)
# dataset=data

dataset['acchange'] = medfilt(dataset['acchange'], 3)              ##Data filtering
dataset.plot()

dataset['diff_1']=dataset['acchange'].diff(1)#First order differencing
dataset['diff_2']=dataset['diff_1'].diff(1)#Second order differencing
dataset.plot(subplots=True,figsize=(18,12))#Data plotting

#
del dataset['diff_2']
#del dataset['acchange']
del dataset['diff_1']
dataset.head()
print(type(dataset))
#Plotting autocorrelation and partial autocorrelation
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(dataset,lags=20,ax=ax1)#autocorrelation 
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout();
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(dataset,lags=20,ax=ax2)#partial autocorrelation
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()


from statsmodels.tsa.stattools import adfuller
fuller_test = adfuller(dataset['acchange'])
fuller_test
#Checking data for stability
def test_p_value(data):
        fuller_test = adfuller(data)
        print('P-value: ',fuller_test[1])
        if fuller_test[1] <= 0.05:
            print('Reject null hypothesis, data is stationary')
        else:
            print('Do not reject null hypothesis, data is not stationary')
            
test_p_value(dataset['acchange'])

            
nu=int(len(dataset)*0.9)#Dividing the data
train = dataset[:nu]
test = dataset[nu:]

model = sm.tsa.ARIMA(train,order=(12,0,3))#5
#model = sm.tsa.ARIMA(train,order=(10,1,8))#10
#model = sm.tsa.ARIMA(train,order=(9,0,4))#11
#model = sm.tsa.ARIMA(train,order=(14,1,3))#15
#model = sm.tsa.ARIMA(train,order=(14,1,4))#14
results = model.fit(disp=1)#Fitting training data
results.summary()
results.resid
plt.figure()
results.resid.plot()
plt.figure()
results.resid.plot(kind='kde')

df_shift = train['acchange'][-1]
forecast1 = results.predict()#Inverse prediction of training data
predict1 = forecast1.add(df_shift)
train= train.reset_index(drop=True)
predict1 = predict1.reset_index(drop=True)
train['prediction'] = predict1 

#train['diff_1']=train['acchange'].diff(1)
# train['prediction']=train['prediction'].apply(lambda x:x-45)
#train['prediction']=train['prediction'].apply(lambda x:x-10)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit_transform(train['prediction'])
train['prediction']=pd.DataFrame(train['prediction'])
train[['acchange','prediction']].plot(figsize=(12,8))#Visualisation of real and predicted data

def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
MSE=metrics.mean_squared_error(train['acchange'], train['prediction']) 
RMSE= metrics.mean_squared_error(train['acchange'], train['prediction'])**0.5
MAE= metrics.mean_absolute_error(train['acchange'], train['prediction']) 
MAPE= mape(train['acchange'], train['prediction'])
r2= metrics.r2_score(train['acchange'], train['prediction']) 
print("MSE  = " + str(round(MSE, 5)))
print("RMSE = " + str(round(RMSE, 5)))
print("MAE  = " + str(round(MAE, 5)))
print("MAPE = " + str(round(MAPE, 5))+'%')
print("R2   = " + str(round(r2, 5)))

delta = train['prediction'] - train['acchange'] # 残差
score = 1 - delta.var()/train['acchange'].var()
print(delta)
print(score)

final = pd.concat([train,test])
t=len(test)+nu

forecast = results.predict(nu,t)
predict = forecast.add(df_shift)

# final = final.reset_index(drop=True)
test= test.reset_index(drop=True)
predict = predict.reset_index(drop=True)
test['prediction'] = predict 
delta = test['prediction']- test['acchange'] # 残差
score = 1 - delta.var()/test['acchange'].var()
test[['acchange','prediction']].plot(figsize=(12,8))

#############
nu=int(len(dataset)*0.9)
train = dataset[:nu]
test = dataset[nu:]
#model = sm.tsa.ARIMA(train,order=(12,1,3))#5
# model = sm.tsa.ARIMA(train,order=(10,1,8))#10
#model = sm.tsa.ARIMA(train,order=(9,0,4))#11
model = sm.tsa.ARIMA(train,order=(14,1,3))#15
#model = sm.tsa.ARIMA(train,order=(14,1,4))#14
#model = sm.tsa.ARIMA(train,order=(9,1,3))
results = model.fit(disp=0)
results.summary()
results.resid
plt.figure()
results.resid.plot()
plt.figure()
results.resid.plot(kind='kde')
final = pd.concat([train,test])
df_shift =train['acchange'][-1]

t=len(test)+nu
forecast = results.predict(nu,t)
predict = forecast.add(df_shift)

test= test.reset_index(drop=True)
predict = predict.reset_index(drop=True)
test['prediction'] = predict 
test[['acchange','prediction']].plot(figsize=(12,8))#14
plt.ylabel("house power consumption",fontsize=16)
plt.xlabel("Home days", fontsize=16)
plt.title("ARIMA Consumption vs. days ",fontsize=16)
plt.show()

def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
MSE=metrics.mean_squared_error(test['acchange'], test['prediction']) 
RMSE= metrics.mean_squared_error(test['acchange'], test['prediction'])**0.5
MAE= metrics.mean_absolute_error(test['acchange'], test['prediction']) 
MAPE= mape(test['acchange'], test['prediction'])
r2= metrics.r2_score(test['acchange'], test['prediction']) 
print("MSE  = " + str(round(MSE, 5)))
print("RMSE = " + str(round(RMSE, 5)))
print("MAE  = " + str(round(MAE, 5)))
print("MAPE = " + str(round(MAPE, 5))+'%')
print("R2   = " + str(round(r2, 5)))

delta = test['prediction']- test['acchange'] # 残差
score = 1 - delta.var()/test['acchange'].var()


###futher 30 days #############
from pandas.tseries.offsets import DateOffset 
date=pd.to_datetime(dataset.index[-1])
extradates = [date + DateOffset(days=n) for n in range (1,30)]

forecast_df = pd.DataFrame(index = extradates,columns = dataset.columns) 
forecast_df.head()
final_df = pd.concat([dataset,forecast_df])

start=len(dataset)
end=start+30
df_shift1 = dataset['acchange'][-1]
forecast1 = results.predict(start, end)
predict1 = forecast1.add(df_shift1)

final_df['prediction'] = predict1 
final_df[['acchange','prediction']].plot(figsize = ( 12,8 ))







