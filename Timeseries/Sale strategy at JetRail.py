import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
pd.set_option('display.float_format', lambda x: '%.5f' % x)

os.chdir("D:/Documents/Education/Data Science ProDegree Imarticus/Python projects/Analytics Vidya and Kaggle/time series jetprop")

imptrain=pd.read_csv("train.csv")
imptrain.drop(["ID"], axis=1, inplace=True)

imptest=pd.read_csv("test.csv")

#we need to remove the time portion

for i in (imptrain, imptest):
    i["Datetime"]=pd.to_datetime(i["Datetime"],format='%d-%m-%Y %H:%M')
    i["Year"]=i["Datetime"].dt.year
    i["month"]=i["Datetime"].dt.month
    i["Day"]=i["Datetime"].dt.day
    i["Hour"]=i["Datetime"].dt.hour


#as the calculations is done on daily basis and the output is required so we need the ratio of number passengers per hour
imptrain["ratio"]=imptrain["Count"]/imptrain["Count"].sum()
ratiohour=imptrain.groupby("Hour")["ratio"].sum()

ratiohour.plot.bar()


#daily rollup

Train_an=imptrain[["Count", "Datetime"]]
Train_an.set_index("Datetime", inplace=True)
Train_an.index.max()-Train_an.index.min()

Test_dt=imptest.set_index("Datetime")

Train_daily=Train_an.resample('D').mean()
Test_Daily=Test_dt.resample('D').mean()



#stationary adfuller test
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
def stationary(df):
    #rolling stats
    rmean=df.rolling(24).mean()
    rolstd=df.rolling(24).std()
    
    
    #plot 01
    print("Plot for RMSE and Rolling Standard Deviation")
    plt.plot(df, color='blue', label='Original')
    plt.plot(rmean, color='red', label='Mean')
    plt.plot(rolstd, color='black', label='STD')
    plt.legend(loc='best')
    plt.title("Rolling stats")
    plt.show()
    
    data1=df.iloc[:,0].values  #function accepts only 1d array of time series so first convert it using
    #plot 
    print("Auto correlation plot")
    plot_acf(data1)
    
    print("ADFuller Test")
    dftest=adfuller(data1, autolag='aic')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistics', 'p-value', 'Lags used', 'no of obs'])
    for key, value in dftest[4].items():
        dfoutput["Critical value (%s)"%key]=value
    print(dfoutput)

  
stationary(Train_daily) #data is not stationary


#converting to stationary
def diff(intr, df):
    diffn=[]
    for pos in range(intr, df.shape[0]):
        val=df.iloc[pos,0]-df.iloc[pos-1,0]
        diffn.append(val)
    avg=sum(diffn[0:5])/5
    diffn.insert(0,avg)
    return diffn

Train_daily_diff=pd.DataFrame(diff(1,Train_daily), index=Train_daily.index, columns=Train_daily.columns)

stationary(Train_daily_diff)

Train=Train_daily[:672]
valid=Train_daily[672:]
#plot.
plt.plot(Train, 'b')
plt.plot(valid, 'g')
plt.show()


#time series decompose
from statsmodels.tsa.seasonal import seasonal_decompose 
decomp=seasonal_decompose(Train_daily_diff)
decomp.plot()

#SES
from statsmodels.tsa.api import Holt, SimpleExpSmoothing, ExponentialSmoothing
SES=SimpleExpSmoothing(Train).fit(smoothing_level=0.1,optimized=False)
forecast=SES.forecast(len(valid)).rename('forecast')
Actual_forecastdf=pd.concat([Train_daily_diff, forecast], axis=1)
#plot
sns.lineplot(data=Actual_forecastdf) 
#validation
validationdf=Actual_forecastdf[-len(valid):].copy()
np.sqrt(np.mean((validationdf["Count"]-validationdf["forecast"])**2))


#DES
DES=Holt(Train).fit(smoothing_level=0.01, smoothing_slope=0.4)
forecast=DES.forecast(len(valid)).rename('forecast')
Actual_forecastdf=pd.concat([Train_daily_diff, forecast], axis=1)
#plot
sns.lineplot(data=Actual_forecastdf)
#validation
validationdf=Actual_forecastdf[-len(valid):]
np.sqrt(np.mean((validationdf["Count"]-validationdf["forecast"])**2))

#TES
TES=ExponentialSmoothing(Train,seasonal_periods=7,seasonal='add', trend='add', damped=False).fit(smoothing_level=0.1, smoothing_slope=0.1)
forecast=TES.forecast(len(valid)).rename('forecast')
Actual_forecastdf=pd.concat([Train_daily_diff, forecast], axis=1)
#plot
sns.lineplot(data=Actual_forecastdf)
#validation
validationdf=Actual_forecastdf[-len(valid):]
np.sqrt(np.mean((validationdf["Count"]-validationdf["forecast"])**2))

#gridsearch
from numpy import arange
griddf=pd.DataFrame()
for sp in ['add']:
    for td in ['add']:
        for sl in arange(0.10, 0.99, .01):
            for slp in arange(0.1, 0.9, 0.1):
                print(sp," ,", td," ,",sl," ,",slp)
                TES=ExponentialSmoothing(Train,seasonal=sp, trend=td).fit(smoothing_level=sl, smoothing_slope=slp)
                forecast=TES.forecast(len(valid)).rename('forecast')
                Actual_forecastdf=pd.concat([Train_daily_diff, forecast], axis=1)
                validationdf=Actual_forecastdf[-len(valid):].copy()
                rmse=np.sqrt(np.mean((validationdf["Count"]-validationdf["forecast"])**2))
                tempdf=pd.DataFrame([[sp,td,sl,slp, rmse]])
                print(rmse)
                griddf=griddf.append(tempdf)
                

#ARIMA
from pmdarima import auto_arima

#get order (p,d,q)lags and seasonal (P,D,Q)
arima1=auto_arima(Train, stationary=False)

arima1.get_params()['order']  #auto set to (1, 1, 2)
arima1.get_params()['seasonal_order'] #autoset to (2, 0, 0, 7)
#forecast
forecast=pd.Series(arima1.predict(len(valid)))
forecast.index=valid.index
forecast.name="forecast"
Actual_forecastdf=pd.concat([Train_daily, forecast], axis=1)

#plot
from matplotlib.pyplot import figure
figure()
sns.lineplot(data=Actual_forecastdf)
#validation
validationdf=Actual_forecastdf[-len(valid):]
np.sqrt(np.mean((validationdf["Count"]-validationdf["forecast"])**2))
arima1.aic()

#MANUAL ARIMA
#
from pmdarima import ARIMA
arima2=ARIMA((8,2,11),(2,1,0,1)).fit(Train)  

#(8, 2, 9)
#(8, 2, 11) (2, 1, 0, 1)

forecast=pd.Series(arima2.predict(len(valid)))
forecast.index=valid.index
forecast.name="forecast"
Actual_forecastdf=pd.concat([Train_daily, forecast], axis=1)
sns.lineplot(data=Actual_forecastdf)
#validation
validationdf=Actual_forecastdf[-len(valid):]
np.sqrt(np.mean((validationdf["Count"]-validationdf["forecast"])**2))

#ARIMA grid search
griddf2=pd.DataFrame()
import itertools
p=range(4,9)
d=[2,3]
q=range(7,12)
pdq=list(itertools.product(p,d,q))
Ps=Ds=Qs=range(0,3)
PsDsQsSs=list(itertools.product(Ps,Ds,Qs))
seasonal=[(x[0], x[1], x[2], 1) for x in PsDsQsSs]
for params in pdq:
    for season_par in seasonal:
        try:
            arima3=ARIMA((params),(season_par)).fit(Train)
            forecast=pd.Series(arima3.predict(len(valid)))
            forecast.index=valid.index
            forecast.name="forecast"
            Actual_forecastdf=pd.concat([Train_daily, forecast], axis=1)
            #validation
            validationdf=Actual_forecastdf[-len(valid):]
            rmse=np.sqrt(np.mean((validationdf["Count"]-validationdf["forecast"])**2))
            aic=arima3.aic()
            print(params," ", season_par," ", rmse," ", aic)
            tempdf=pd.DataFrame([[params,season_par,rmse, aic]])
            griddf2=griddf2.append(tempdf)
        except:
            continue
    
griddf2.to_csv("arima_grid.csv")
   
#prediction ARIMA
forecast=pd.Series(arima2.predict(len(Test_Daily)))
forecast.index=Test_Daily.index
forecast.name="forecast"
Test_Daily["pred"]=forecast
merge1=pd.merge(Test_Daily,imptest, on=("Year", "month", "Day"), how="left")
merge1["Hour"]=merge1["Hour_y"]
prediction=pd.merge(merge1, ratiohour, on="Hour", how="left")
prediction["Count"]=prediction["pred"]*prediction["ratio"]*24
submission=pd.DataFrame({"ID":prediction["ID_y"], "Count":prediction["Count"]})
submission.to_csv("Arima_(5,1,8,1,1,0,1).Dailyconvo2.csv", index=False)



#Trial prediction using DES/SES/TES
forecast=TES.forecast(len(Test_Daily)).rename('forecast')
forecast.index=Test_Daily.index
forecastdf=pd.DataFrame(forecast)

#integration
def intr(intr, df):
    intrn=[]
    for pos in range(intr, df.shape[0]):
        val=np.where(df.iloc[pos,0]<0,np.abs(df.iloc[pos,0]),df.iloc[pos,0])+np.where(df.iloc[pos-1,0]<0,np.abs(df.iloc[pos-1,0]),df.iloc[pos-1,0])
        intrn.append(val)
    avg=sum(intrn[0:5])/5
    intrn.insert(0,avg)
    return intrn

forecastint=intr(1, forecastdf)


Test_Daily["pred"]=forecastint
merge1=pd.merge(Test_Daily,imptest, on=("Year", "month", "Day"), how="left")
merge1["Hour"]=merge1["Hour_y"]
prediction=pd.merge(merge1, ratiohour, on="Hour", how="left")
prediction["Count"]=prediction["pred"]*prediction["ratio"]*24
submission=pd.DataFrame({"ID":prediction["ID_y"], "Count":prediction["Count"]})
submission.to_csv("TES_diff.csv", index=False)


