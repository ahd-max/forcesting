# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 13:44:46 2021

@author: Hangdong AN
## Data processing and model building for Randomforest regrassion models as well as data testing and visualisation of results.
"""
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
df['temp']=df['temp'].apply(lambda x:x-273.15)
df['2mtemp']=df['2mtemp'].apply(lambda x:x-273.15)
df= df.drop(df[df['PV_gen'] > 30].index)
df= df.drop(df[df['PV_gen'] < 0].index)
sn=df['SerialNumber'].unique()
df= df.drop(df[df['AC_consumption'] > 40].index)
df['rain']=df['rain'].apply(lambda x: 1 if x > 0.1 else 0)
df=df.drop(['Unnamed: 0'],axis=1)

dfd=df[['UTC_time','PV_gen','temp','2mtemp','AC_consumption','weekday','Season']]
dff=df[['UTC_time','PV_gen','temp','AC_consumption','weekday']]
import seaborn as sns

sns.pairplot(dfd,hue='AC_consumption')
plt.savefig('all0',dpi=1800)


sns.pairplot(dff,hue='AC_consumption')
plt.savefig('all1',dpi=1800)
newdata=pd.DataFrame()
X_train1=pd.DataFrame()
y_train1=pd.DataFrame()
X_test1=pd.DataFrame()
y_test1=pd.DataFrame()

def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

A=[]
from sklearn.model_selection import train_test_split
for i in range(len(sn)):
     A=sn[i]
     newdata=df[df['SerialNumber'] == A]
     if len(newdata) == 0:
         continue
         
     else:
         X = newdata[['SerialNumber','PV_gen','temp','2mtemp','rain','Season','weekday']]
         y = newdata[['SerialNumber','AC_consumption']]
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
         X_train1=X_train1.append(X_train,ignore_index=False)
         y_train1=y_train1.append(y_train,ignore_index=False)
         X_test1=X_test1.append(X_test,ignore_index=False)
         y_test1=y_test1.append(y_test,ignore_index=False)


# ,'home'
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100,max_features='auto',random_state=101,
#                                criterion='gini', max_depth=None,min_samples_split=2,
#                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
#                                n_jobs=1)
# X_train1=X_train1[['PV_gen','temp','2mtemp','rain','Season','weekday','home']]
# y_train1=y_train1[['AC_consumption']]


# X_train1 = X_train1.fillna(X_train1.mean())
# y_train1 = y_train1.fillna(y_train1.mean())
# model.fit(X_train1,y_train1.astype('int'))


# user=X_test1['SerialNumber'].unique()

# for i in range(8):
    
#     X_test=X_test1[X_test1['SerialNumber'] == user[i]]
#     y_test=y_test1[y_test1['SerialNumber'] == user[i]]
#     X_test=X_test[['PV_gen','temp','2mtemp','rain','Season','weekday','home']]
#     y_test=y_test[['AC_consumption']]

# #forcasting
#     from sklearn.metrics import accuracy_score
#     x=np.array(len(y_test))
#     preds = model.predict(X_test)
#     accuracy=accuracy_score(y_test,preds)
#     print("Accuracy : %.2f%%" % (accuracy * 100.0))
#     from sklearn.metrics import classification_report

#     report = classification_report(y_test, preds, output_dict=True)
#     print(classification_report(preds, y_test))

# #d0 = pd.DataFrame(score).transpose()
#     d1 = pd.DataFrame(report).transpose()

#     res = pd.concat([d1], axis=1)
#     rawdata=rawdata.append(res,ignore_index=False)
# rawdata.to_csv("result.csv", index= True, mode = 'w')


# 训练随机森林解决回归问题
from sklearn.ensemble import RandomForestRegressor
X_train1=X_train1[['PV_gen','temp','2mtemp','rain','Season','weekday']]
y_train1=y_train1[['AC_consumption']]
regressor = RandomForestRegressor(n_estimators=100, random_state=100)
regressor.fit(X_train1, y_train1)

a=df[['temp']]
b=df[['2mtemp']]

fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=plt.plot(a,lags=20,ax=ax1)#自相关
ax1.xaxis.set_ticks_position('bottom')

ax2=fig.add_subplot(212)
fig=plt.plot(b,lags=20,ax=ax1)#偏自相关
ax2.xaxis.set_ticks_position('bottom')




import seaborn as sns
sns.pairplot(df,hue='AC_consumption')

da=df[['PV_gen','temp','AC_consumption','weekday','Season']]
sns.pairplot(da,hue='AC_consumption')
# sns.heatmap(da,annot=True)

x=da['temp']
y=da['AC_consumption']
sns.regplot(x,y)

aa=df[['temp','AC_consumption']]
aa=aa.corr()
sns.heatmap(aa,nnot=True,cmap='Reds')

#




importances = regressor.feature_importances_
#indices = np.argsort(importances)[::-1]

feat_labels = X_train1.columns[0:]
imp=[]
for f in range(X_train1.shape[1]):
    print(f + 1, feat_labels[f], importances[f])
##plot
c=regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_],
              axis=0)
indices = np.argsort(c)[::-1]
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train1.shape[1]), c[indices],
        color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train1.shape[1]), indices)
plt.xlim([-1, X_train1.shape[1]])
plt.show()


rawdata=pd.DataFrame()
from sklearn.metrics import roc_auc_score
R2 =[]
day=[]
AUC=[]
user=X_test1['SerialNumber'].unique()

for i in range(len(user)):
    X_test=X_test1[X_test1['SerialNumber'] == user[i]]
    y_test=y_test1[y_test1['SerialNumber'] == user[i]]
    X_test=X_test[['PV_gen','temp','rain','Season','weekday']]
    y_test=y_test[['AC_consumption']]
#forcasting
    y_pred = regressor.predict(X_test)
# 评估回归性能
    from sklearn import metrics
    # MAPE = mape(y_test, y_pred)
    r2   = metrics.r2_score(y_test, y_pred) 
    b=len(y_test)
    R2.append(r2)
    day.append(b) 
    auc= roc_auc_score(y_test,y_pred)
    AUC.append(auc)
    # mapd.append(MAPE)
R2=pd.DataFrame(R2)
day=pd.DataFrame(day)

# result = pd.concat([day,R2], axis=1)
# result.to_csv("acpr.csv", index= True, mode = 'w')


import numpy as np
R2=np.array(R2)
c=np.sum(R2>=0)                   
d=np.sum(R2<0)
print ("大于等于0的个数:  "+str(c)) #输出满足条件的个数
print ("小于0的个数:  "+str(d))
y = np.array([32,68])

plt.pie(y,
        labels=['R2 better','R2 poor'], # 设置饼图标签
        colors=["#d5695d", "#5d8ca8"] # 设置饼图颜色
       )
plt.title("R2 distribution of different households in RFR model",fontsize=16) # 设置标题
plt.show()

poordata=pd.DataFrame()
poor=pd.DataFrame()
gooddata=pd.DataFrame()
good=pd.DataFrame()

for i in range(len(R2)):
    if R2[i]>0:
        gooddata=X_test1[X_test1['SerialNumber'] == user[i]]
        good=good.append(gooddata,ignore_index=False)

for i in range(len(R2)):
    if R2[i]<0:
        poordata=X_test1[X_test1['SerialNumber'] == user[i]]
        poor=poor.append(poordata,ignore_index=False)
        
        
X_train1['test']=0
good['test']=1
poor['test']=2
# df_adv = pd.concat([X_train1, good])#0.5568273133856805
df_adv = pd.concat([X_train1, poor])#0.5136054154964397
x = df_adv.drop( [ 'test' ,'SerialNumber'], axis = 1 )
y = df_adv['test']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
linreg.fit(x_train,y_train)
y_pred = linreg.predict(x_test)
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_test,y_pred)
print(auc_score)


# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators=100, random_state=100)
# regressor.fit(x_train,y_train)
# y_pred = regressor.predict(x_test)
# from sklearn.metrics import roc_auc_score
# auc_score = roc_auc_score(y_test,y_pred)
# print(auc_score)





for i in range(len(R2)):
    if R2[i]<0:
        poordata=X_test1[X_test1['SerialNumber'] == user[i]]
    elif R2[i]>0:
        gooddata=X_test1[X_test1['SerialNumber'] == user[i]]
    else:
        continue

x=result1['0']
y=result1['0.1']
plt.scatter(x,y)
plt.ylabel("R2",fontsize=16)
plt.xlabel("Home days", fontsize=16)
plt.title("Volume of data vs. R2 ")
plt.ylim(-1, 1)

plt.show()
plt.bar(x,y)



    
# for i in range(len(user)):














R2 =[]
day=[]

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
X_train1=X_train1[['PV_gen','temp','rain','Season','weekday']]
y_train1=y_train1[['AC_consumption']]
linreg.fit(X_train1,y_train1)

rawdata=pd.DataFrame()

user=X_test1['SerialNumber'].unique()

for i in range(len(user)):
# for i in range(1):
    X_test=X_test1[X_test1['SerialNumber'] == user[i]]
    y_test=y_test1[y_test1['SerialNumber'] == user[i]]
    X_test=X_test[['PV_gen','temp','rain','Season','weekday']]
    y_test=y_test[['AC_consumption']]

#forcasting
    from sklearn.metrics import accuracy_score
    x=np.array(len(y_test))
    y_pred = linreg.predict(X_test)

    from sklearn                import metrics
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    RMSE = metrics.mean_squared_error(y_test, y_pred)**0.5
    MSE  = metrics.mean_squared_error(y_test, y_pred) 
    MAE  = metrics.mean_absolute_error(y_test, y_pred)
    r2   = metrics.r2_score(y_test, y_pred) 
    b=len(y_test)
    R2.append(r2)
    day.append(b)
    # mapd.append(MAPE)
    
R2=pd.DataFrame(R2)
day=pd.DataFrame(day)  
    
result1 = pd.concat([day,R2], axis=1)
result1.columns = ['days','R2']
x=result1['days']
y=result1['R2']
plt.scatter(x,y)
plt.ylabel("R2",fontsize=16)
plt.xlabel("Home days", fontsize=16)
plt.title("Volume of data vs. R2 ")
plt.ylim(-1, 1)

plt.show()
plt.bar(x,y)

print("RMSE = " + str(round(RMSE, 5)))
print("MSE  = " + str(round(MSE, 5)))
print("MAE  = " + str(round(MAE, 5)))














