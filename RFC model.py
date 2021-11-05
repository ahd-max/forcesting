import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil import tz
import csv
import time
import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv('pingjunzhi.csv')
df.head()
df['temp'] = df['temp'].apply(lambda x: x - 273.15)
df['2mtemp'] = df['2mtemp'].apply(lambda x: x - 273.15)
df = df.drop(df[df['PV_gen'] > 30].index)
df = df.drop(df[df['PV_gen'] < 0].index)
sn = df['SerialNumber'].unique()
df = df.drop(df[df['AC_consumption'] > 40].index)
df['rain'] = df['rain'].apply(lambda x: 1 if x > 0.1 else 0)
df = df.drop(['Unnamed: 0'], axis=1)

# dd = df.groupby('SerialNumber')

newdata = pd.DataFrame()
X_train1 = pd.DataFrame()
y_train1 = pd.DataFrame()
X_test1 = pd.DataFrame()
y_test1 = pd.DataFrame()

A = []
from sklearn.model_selection import train_test_split

for i in range(len(sn)):
    A = sn[i]
    newdata = df[df['SerialNumber'] == A]
    if len(newdata) == 0:
        continue

    else:
        X = newdata[['SerialNumber', 'AC_consumption', 'PV_gen', 'temp', '2mtemp', 'rain', 'Season', 'weekday']]
        y = newdata[['SerialNumber', 'home']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        X_train1 = X_train1.append(X_train, ignore_index=False)
        y_train1 = y_train1.append(y_train, ignore_index=False)
        X_test1 = X_test1.append(X_test, ignore_index=False)
        y_test1 = y_test1.append(y_test, ignore_index=False)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, max_features='auto', random_state=101,
                               criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                               n_jobs=1)
X_train1 = X_train1[['AC_consumption', 'PV_gen', 'temp', '2mtemp', 'rain', 'Season', 'weekday']]
y_train1 = y_train1[['home']]

X_train1 = X_train1.fillna(X_train1.mean())
y_train1 = y_train1.fillna(y_train1.mean())
model.fit(X_train1, y_train1.astype('int'))

# 基于随机森林度量各个变量的重要性
importances = model.feature_importances_
# indices = np.argsort(importances)[::-1]

feat_labels = X_train1.columns[0:]
imp = []
for f in range(X_train1.shape[1]):
    print(f + 1, feat_labels[f], importances[f])
##plot
c = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(c)[::-1]
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train1.shape[1]), c[indices],
        color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train1.shape[1]), indices)

plt.xlim([-1, X_train1.shape[1]])
plt.show()
rawdata = pd.DataFrame()

user = X_test1['SerialNumber'].unique()

for i in range(len(user)):
    X_test = X_test1[X_test1['SerialNumber'] == user[i]]
    y_test = y_test1[y_test1['SerialNumber'] == user[i]]
    X_test = X_test[['AC_consumption', 'PV_gen', 'temp', '2mtemp', 'rain', 'Season', 'weekday']]
    y_test = y_test[['home']]

    # forcasting
    from sklearn.metrics import accuracy_score

    x = np.array(len(y_test))
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print("Accuracy : %.2f%%" % (accuracy * 100.0))
    from sklearn.metrics import classification_report

    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(preds, y_test))

    # d0 = pd.DataFrame(score).transpose()
    d1 = pd.DataFrame(report).transpose()

    res = pd.concat([d1], axis=1)
    rawdata = rawdata.append(res, ignore_index=False)
rawdata.to_csv("result.csv", index=True, mode='w')

