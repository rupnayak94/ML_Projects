import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing
from pmdarima import auto_arima, ARIMA
import itertools
from sklearn.impute import KNNImputer


raw_data=pd.read_csv("GlobalLandTemperaturesByCountry.csv")
raw_i=raw_data[raw_data["Country"]=="India"].copy()

raw_i.drop(["AverageTemperatureUncertainty", "Country"],axis=1, inplace=True)

raw_i.info()
raw_i["dt"]
raw_i["dt"]=pd.to_datetime(raw_i["dt"], format="%Y-%m-%d")

raw_i.set_index("dt", inplace=True)
plt.plot(raw_i)

raw_i=raw_i.resample("Y").mean()

na=raw_i.isnull().sum().item()
while(na>0):
    print(na)
    raw_i.fillna(raw_i.rolling(12, min_periods=1).mean(), inplace=True)
    na=raw_i.isnull().sum().item()
plt.plot(raw_i)

train=raw_i["1796-01-01":"2000-12-31"].copy()
test=raw_i["2001-01-01":].copy()

train.isnull().sum()
test.isnull().sum()



#ploting
%matplotlib notebook
plt.plot(train, "g")
plt.plot(test, "b")
plt.ylabel("avg temperature")
plt.xlabel("year")
plt.show()



#stationary check
def stationary(df):
    m_avg=df.rolling(12).mean()
    m_std=df.rolling(12).std()
    
    #plots to check stationality
    %matplotlib inline
    plt.ylabel("avg temperature")
    plt.xlabel("year")
    plt.plot(df, "g", label="data")
    plt.plot(m_avg, "b", label="mean")
    plt.plot(m_std, "r", label="std")
    plt.legend(loc="best")
    plt.show()
    
    #acf plot
    %matplotlib notebook
    plot_acf(df)
    plot_pacf(df)
    
    #adfuller test
    st_test=adfuller(df)
    st_test_sr=pd.Series(st_test, index=["T-test", "Pvalue", "Lags used", 
                                         "no of obs", "Critical values", "infm_ctr"])
    return st_test_sr

sttn=stationary(raw_i) #the data found to be stationnary

"""
the data is stationary as:-

Adfuller test:

T-test = -3.50093 which is less than 
Critical values   {'1%': -3.4471856790801514, '5%': -2.868960436182993, '10%': -2.5707229006220524

also the Pvalue is < than 0.05 (pvalue is 0.0079) thus we reject the null hypo that the data in
not stationary.

PLot_ACF
the delay is also rapid so the data is stationary


"""

#decomposition
decom=seasonal_decompose(raw_i)
%matplotlib inline
decom.plot()

#Simple exponential smooothing

SES=SimpleExpSmoothing(train).fit()
forecast=SES.forecast(len(test)).rename("forecast")
Climate_org_for=pd.concat([fullraw_i19, forecast], axis=1)
plt.plot(Climate_org_for)
validationdf=Climate_org_for[-len(test):]
rmse=np.sqrt(mean_squared_error(validationdf["AverageTemperature"], validationdf["forecast"]))
SES.params
SES.aic

#Double Exponential Smoothing

DES=Holt(train).fit()
forecast=DES.forecast(len(test)).rename("forecast")
Climate_org_for=pd.concat([fullraw_i19, forecast], axis=1)
plt.plot(Climate_org_for)
validationdf=Climate_org_for[-len(test):]
rmse=np.sqrt(mean_squared_error(validationdf["AverageTemperature"], validationdf["forecast"]))
DES.params
DES.aic

#Triple Exponential smoothing
TES=ExponentialSmoothing(train, trend="add", seasonal="add", 
                         seasonal_periods=12).fit(smoothing_level=0.7)
forecast=TES.forecast(len(test)).rename("forecast")
Climate_org_for=pd.concat([fullraw_i19, forecast], axis=1)
%matplotlib inline
plt.plot(Climate_org_for)
validationdf=Climate_org_for[-len(test):]
rmse=np.sqrt(mean_squared_error(validationdf["AverageTemperature"], validationdf["forecast"]))
TES.params
TES.aic


#USING auto ARIMA
auto_a=auto_arima(train,d=1, m=12, suppress_warnings=True, trace=True, n_jobs=-1)
auto_a.get_params()
auto_a.summary()
forecast=pd.Series(auto_a.predict(len(test)))
forecast.index=test.index
forecast.name="forecast"
Climate_org_for=pd.concat([raw_i, forecast], axis=1)
plt.plot(Climate_org_for)
validationdf=Climate_org_for[-len(test):]
rmse=np.sqrt(mean_squared_error(validationdf["AverageTemperature"], validationdf["forecast"]))


#grid seacg with manual arima


#defining parameters

"""
the itertool is used to create an "cartesian product, equivalent to a nested for-loop" 
or each one is combined with each one to form a final set.)
"""

p=q=range(1,4)
d=range(1,3)
pdq=list(itertools.product(p,d,q))

Ps=Qs=range(1,4)
Ds=[1,2]
Ss=[12]
PsDsQs=list(itertools.product(Ps, Ds, Qs, Ss))
count=(len(pdq)*len(PsDsQs))
griddf=pd.DataFrame()
for arima_par in pdq:
    for seasonal_par in PsDsQs:
        try:
            print("Iteration: ",count," :::****************************************")
            m_arima=ARIMA((arima_par),(seasonal_par), suppress_warnings=True).fit(train)
            forecast=pd.Series(m_arima.predict(len(test)))
            forecast.index=test.index
            forecast.name="forecast"
            Climate_org_for=pd.concat([raw_i, forecast], axis=1)
            validationdf=Climate_org_for[-len(test):]
            rmse=np.sqrt(mean_squared_error(validationdf["AverageTemperature"], validationdf["forecast"]))
            aic=m_arima.aic()
            temp=pd.Series({"pqd":arima_par, "Seasonal":seasonal_par, "RMSE":rmse, "aic":aic})
            print(temp)
            griddf=griddf.append(temp, ignore_index=True)
            count-=1
        except:
            continue
            
griddf.to_csv("gridcv.csv")

grid_frm_csv=pd.read_csv("gridcv.csv")

mm_arima=ARIMA((1,1,1),(2,1,3,12), suppress_warnings=True).fit(train)
forecast=pd.Series(mm_arima.predict(len(test)))
forecast.index=test.index
forecast.name="forecast"
Climate_org_for=pd.concat([raw_i, forecast], axis=1)
validationdf=Climate_org_for[-len(test):]
rmse=np.sqrt(mean_squared_error(validationdf["AverageTemperature"], validationdf["forecast"]))
aic=mm_arima.aic()
%matplotlib notebook
plt.plot(Climate_org_for)








