# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 12:45:40 2021

@author: Hangdong AN
@e-mail：hangdong.an@uqconnect.edu.au
## Data processing and model building for LSTM models as well as data testing and visualisation of results.

"""
import pandas            as pd
import matplotlib.pyplot as plt
import tensorflow        as tf 
import numpy             as np
from tensorflow.keras.models          import Sequential
from tensorflow.keras.layers          import Dense,LSTM,Bidirectional

from numpy                 import array
from sklearn                import metrics
from sklearn.preprocessing import MinMaxScaler


# Data reconciliation
from scipy.ndimage         import gaussian_filter1d
from scipy.signal          import medfilt

# Results reproduced
from numpy.random          import seed

n_epochs     = 50    # Training theory numbers
filter_on    = 1     # Activation filters

model_type = 1
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

data = pd.read_csv('33.csv') #Identifying documents
ID2=data['SerialNumber'].unique()
data=data[['SystemSnapshotDate_UTC','SerialNumber','acchange']]
data=data[data['SerialNumber'] == ID2[5]]#File data selection Select family
data=data[['SystemSnapshotDate_UTC','acchange']]

dataset_new = data
for index, row in data.iterrows():
    dataset_new["SystemSnapshotDate_UTC"][index] = row["SystemSnapshotDate_UTC"].split(" ")[0]#Data aggregation
#print(dataset_new.head())
dataset_new_2 = dataset_new.groupby(by='SystemSnapshotDate_UTC')['acchange'].sum()*0.00001#Data reduction

dict_dataset = {'SystemSnapshotDate_UTC':dataset_new_2.index,'acchange':dataset_new_2.values}
dataset_new_3 = pd.DataFrame(dict_dataset)
#print(dataset_new_3.head())
dataset = dataset_new_3
fig,ax=plt.subplots(figsize=(6.4,4.8), dpi=100)
ax.plot(dataset['acchange'], label='before filter',color='b', )

# plt.plot(dataset['acchange'],color='b')
# plt.show()
# plt.figure()
if filter_on == 1:  # 数据集过滤
    dataset['acchange'] = medfilt(dataset['acchange'], 3)              # Median filter
    # dataset['acchange'] = gaussian_filter1d(dataset['acchange'], 1.2)  # 高斯过滤
# plt.plot(dataset['acchange'],color='red',linestyle='--')
# plt.ylabel("day power",fontsize=16)
# plt.xlabel("day",fontsize=16)
# plt.title("AC_consumption per day",fontsize=20)
# plt.show()

#Plotting data before and after filtering
ax.plot(dataset['acchange'], label='after filter',color='r', linestyle='--',  )
ax.set_xlabel('day', fontsize=16)
ax.set_ylabel('AC_consumption kwh*0.00001', fontsize=16)
ax.tick_params(axis='both', labelsize=11)
ax.yaxis.grid(True, linestyle='-.')
legend = ax.legend(loc='best')
plt.show()



n_timestamp  = 24    # 时间戳
train_days   = int(len(dataset['acchange'])*0.9)  # Divide training days
testing_days = int(len(dataset['acchange'])*0.1 ) # Number of test days

# Setting up training and test datasets
train_set    = dataset[0:train_days].reset_index(drop=True)
test_set     = dataset[train_days: train_days+testing_days].reset_index(drop=True)
training_set = train_set.iloc[:, 1:2].values
testing_set  = test_set.iloc[:, 1:2].values

# Normalise data in the range 0 to 1
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled  = sc.fit_transform(testing_set)

# The data for the first n_timestamp days is X; the data for n_timestamp+1 days is Y。
def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        
        if end_ix > len(sequence)-1:
            break
            
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

#Data segmentation
X_train, y_train = data_split(training_set_scaled, n_timestamp)
X_train          = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test, y_test   = data_split(testing_set_scaled, n_timestamp)
X_test           = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#LSTM model
if model_type == 1:
    model = Sequential()
    model.add(LSTM(units=50, activation='relu',
                   input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1))
    
#print(model.summary() )
# Model training, the larger the batch_size the more accurate, the more training consumption
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=32)
loss    = history.history['loss']
epochs  = range(len(loss))

# Get the predicted values for the test set dataset
y_predicted = model.predict(X_test)

# Restore the data
y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled     = sc.inverse_transform(y_train)
y_test_descaled      = sc.inverse_transform(y_test)
y_pred               = y_predicted.ravel()           # Converting multidimensional arrays to one-dimensional arrays
y_pred               = [round(i, 2) for i in y_pred] # Retain two decimal places
y_tested             = y_test.ravel()                # Converting multidimensional arrays to one-dimensional arrays

# Make model predictions
y_predicted = model.predict(X_test)

# Show forecast results
plt.figure(figsize=(8, 7))

plt.subplot(3, 2, 3)
plt.plot(y_test_descaled, color='black', linewidth=1, label='True value')
plt.plot(y_predicted_descaled, color='red',  linewidth=1, label='Predicted')
plt.legend(frameon=False)
plt.ylabel("AC comsumption",fontsize=16)
plt.xlabel("Day",fontsize=16)
plt.title("LSTM Predicted data",fontsize=16)

plt.subplot(3, 2, 4)
plt.plot(y_test_descaled[:5], color='black', linewidth=1, label='True value')
plt.plot(y_predicted_descaled[:5], color='red', label='Predicted')
plt.legend(frameon=False)
plt.ylabel("AC comsumption")
plt.xlabel("Day")
plt.title("Predicted data (first few days)")

plt.subplot(3, 3, 7)
plt.plot(epochs, loss, color='black')
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Training curve")

plt.subplot(3, 3, 8)
plt.plot(y_test_descaled-y_predicted_descaled, color='black')
plt.ylabel("Residual")
plt.xlabel("Day")
plt.title("Residual plot")

plt.subplot(3, 3, 9)
plt.scatter(y_predicted_descaled, y_test_descaled, s=2, color='black')
plt.ylabel("Y true")
plt.xlabel("Y predicted")
plt.title("Scatter plot")

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()

# Define your own MAPE function
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

MSE  = metrics.mean_squared_error(y_test_descaled, y_predicted_descaled)      # MSE
RMSE = metrics.mean_squared_error(y_test_descaled, y_predicted_descaled)**0.5
MAE  = metrics.mean_absolute_error(y_test_descaled, y_predicted_descaled)     # MAE
MAPE = mape(y_test_descaled, y_predicted_descaled)
r2   = metrics.r2_score(y_test_descaled, y_predicted_descaled)                  # Coefficient of determination (goodness of fit) closer to 1 is better


print("MSE  = " + str(round(MSE, 5)))
print("RMSE = " + str(round(RMSE, 5)))
print("MAE  = " + str(round(MAE, 5)))
print("MAPE = " + str(round(MAPE, 5)))
print("R2   = " + str(round(r2, 5)))
