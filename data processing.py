"""
Created on Sun Oct  3 22:15:59 2021

@author: Hangdong AN
@e-mail：hangdong.an@uqconnect.edu.au
#Data processing
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
import os
from dateutil import tz
import cdsapi

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'skin_temperature',
        ],
        'year': [
            '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': '13:00',
        'area': [
            -16.01, 112.01, -44.05,
            178,
        ],

    },
    'AU.nc')

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '2m_temperature',
        ],
        'year': [
            '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': '13:00',
        'area': [
            -16.01, 112.01, -44.05,
            178,
        ],

    },
    'AU1.nc')

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'total_precipitation',
        ],
        'year': [
            '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': '13:00',
        'area': [
            -16.01, 112.01, -44.05,
            178,
        ],

    },
    'AU2.nc')

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil import tz
import csv
import time
import datetime

location = pd.DataFrame()
location = pd.read_csv('loc500.csv')
ID1 = location['SerialNumber']
lat = location['Latitude']
lon = location['Longitude']
lat = pd.DataFrame(lat)
lon = pd.DataFrame(lon)

rawdata = pd.DataFrame()

# main processing

data = pd.read_csv('33.csv')
data.head()
# data=pd.read_csv('aend.csv')
ID2 = data['SerialNumber'].unique()

loc = pd.DataFrame()
A = []
for i in range(len(ID2)):  # 得到location
    A = ID2[i]
    a = location[location['SerialNumber'] == A]
    loc = loc.append(a, ignore_index=False)
loc.head()
ID3 = loc['SerialNumber']
lat = loc['Latitude']
lon = loc['Longitude']
lat = pd.DataFrame(lat)
lon = pd.DataFrame(lon)
coords = pd.concat([lat, lon], axis=1)
coords = np.array(coords)
# print(coords)
coord = []
for i in range(len(lat)):
    str1 = ','.join([str(x) for x in coords[i]])
    coord.append(str1)

for i in range(len(ID2)):
    # for i in range(2):
    # data=pd.read_csv('aend.csv')
    datanew = pd.read_csv('33.csv')
    datanew = datanew[datanew['SerialNumber'] == ID2[i]]

    datanew = pd.DataFrame(datanew)
    ac = datanew['acchange']
    ac[ac < 0] = 0
    ac[ac > 18.4] = 0
    lo = coords[i]
    dataset_new = datanew

    # n=dataset_new[dataset_new['SerialNumber'] == ID2[i]]
    # n=dataset_new['SerialNumber']

    ##hourly data to daily data
    for index, row in datanew.iterrows():
        dataset_new["SystemSnapshotDate_UTC"][index] = row["SystemSnapshotDate_UTC"].split(" ")[0]

    ##delete the power comsumtaion less than 0
    dataset_new = dataset_new.drop(dataset_new[dataset_new['acchange'] < 0].index)
    dataset_new['SystemSnapshotDate_UTC'] = pd.to_datetime(dataset_new['SystemSnapshotDate_UTC'])
    datasetset = dataset_new.groupby('SystemSnapshotDate_UTC')['acchange'].sum()
    datasetset1 = dataset_new.groupby('SystemSnapshotDate_UTC')['pvchange'].sum()
    dict_dataset = {'SystemSnapshotDate_UTC': datasetset.index, 'acchange': datasetset.values,
                    'pvchange': datasetset1.values,
                    }
    dataset_new_3 = pd.DataFrame(dict_dataset)

    z = len(dataset_new_3['acchange'])
    n = pd.Series(ID2[i])
    k = pd.DataFrame()

    for j in range(z):
        k = k.append(n, ignore_index=True)

    dataset_new_3 = pd.concat([dataset_new_3, k], axis=1)
    dataset_new_3.columns = ['SystemSnapshotDate_UTC', 'acchange', 'pvchange', 'SerialNumber']

    data = dataset_new_3
    data.loc[:, 'SystemSnapshotDate_UTC'] = pd.to_datetime(data['SystemSnapshotDate_UTC'], format='%Y/%m/%d %H:%M:%S')
    ##define the start time and end time of store

    ##difine the start and end date
    starttime = data['SystemSnapshotDate_UTC'].iloc[0]
    starttime = starttime.strftime('%Y-%m-%d')
    endtime = data['SystemSnapshotDate_UTC'].iloc[-1]
    endtime = endtime.strftime('%Y-%m-%d')

    data = data[(data['SystemSnapshotDate_UTC'] >= pd.to_datetime(starttime)) & (
                data['SystemSnapshotDate_UTC'] <= pd.to_datetime(endtime))]

    ##defind workingday and weekend

    data.loc[:, 'dow'] = data['SystemSnapshotDate_UTC'].apply(lambda x: x.dayofweek)
    data.loc[(data['dow'] < 5), 'weekday'] = 0
    data.loc[(data['dow'] > 4), 'weekday'] = 1
    data.loc[(data['dow'] == 5), 'weekday'] = 1
    data.loc[(data['dow'] == 6), 'weekday'] = 1
    zo = data['dow']
    workday = pd.DataFrame(data[(data['weekday'] == 0)])
    weekend = pd.DataFrame(data[(data['weekday'] == 1)])
    mean_workday = np.mean(workday)
    # data['Date'] = pd.to_datetime(data['Date'])
    data['SystemSnapshotDate_UTC'] = pd.to_datetime(data['SystemSnapshotDate_UTC'])
    data['month'] = data['SystemSnapshotDate_UTC'].dt.month
    # mean_workday=mean_workday['acchange'].iloc[0]
    data.loc[:, 'Season'] = data['month'].apply(lambda x:
                                                1 if x in [9, 10, 11]
                                                else (2 if x in [12, 1, 2]
                                                      else (3 if x in [3, 4, 5]
                                                            else (4 if x in [6, 7, 8] else 0))))
    ## 1 ==spring, 2==summer, 3==autumn, 4==winter
    mean_weekend = np.mean(weekend)
    # h=data['weekday']
    h = data['Season']

    g = []
    for i in range(len(zo)):
        g = h[i]
        if g == 0:
            data.loc[:, 'on/off'] = data['acchange'].apply(lambda x: 1 if x > mean_workday[0] else 0)
        else:
            data.loc[:, 'on/off'] = data['acchange'].apply(lambda x: 1 if x > mean_weekend[0] else 0)

    ## Matching weather data with energy data added
    import xarray as xr

    temp = xr.open_dataset('AU.nc')
    mskt = xr.open_dataset('AU1.nc')
    rain = xr.open_dataset('AU2.nc')

    y = lo[0]
    x = lo[1]

    tem = temp.sel(time=slice(starttime, endtime))
    tem = tem['skt'].sel(longitude=x, latitude=y, method='nearest')
    tem = tem.to_dataframe()
    skt = tem['skt']
    skt = np.array(skt)
    skt = pd.DataFrame(skt)

    msk = mskt.sel(time=slice(starttime, endtime))
    msk = msk['t2m'].sel(longitude=x, latitude=y, method='nearest')
    msk = msk.to_dataframe()
    msk = msk['t2m']
    msk = np.array(msk)
    msk = pd.DataFrame(msk)

    rai = rain.sel(time=slice(starttime, endtime))
    rai = rai['tp'].sel(longitude=x, latitude=y, method='nearest')
    rai = rai.to_dataframe()
    rai = rai['tp']
    rai = np.array(rai)
    rai = pd.DataFrame(rai)

    b = data['SerialNumber']
    p = data['pvchange']
    q = data['acchange']
    q1 = data['SystemSnapshotDate_UTC']
    q2 = data['dow']
    home = data['on/off']
    Season = data['Season']
    q = pd.DataFrame(q)
    q1 = pd.DataFrame(q1)
    q2 = pd.DataFrame(q2)
    home = pd.DataFrame(home)
    Season = pd.DataFrame(Season)
    result = pd.concat([q1, b, q, p, skt, msk, rai, q2, Season, home], axis=1)
    rawdata = rawdata.append(result, ignore_index=False)

rawdata.head()
filename = 'pingjunzhi.csv'
rawdata.to_csv(filename,
               header=['UTC_time', 'SerialNumber', 'AC_consumption', 'PV_gen', 'temp', '2mtemp', 'rain', 'weekday',
                       'Season', 'home'], mode='w')
