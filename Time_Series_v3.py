# importing library

from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import pandas as pd
import os
import seaborn as sns
import numpy as np

# directory change
os.chdir("C:\\Users\\ASUS\\Desktop\\DataSet")
#FullRaw = pd.read_csv("Super Store.csv")
FullRaw=pd.read_csv("SuperStore.csv",encoding="ISO-8859-1")

###############
# Check for NA values
###############

FullRaw.isnull().sum()
FullRaw.head()

###############
# Select ONLY needed columns and roll-up (sum) the data at a daily level!
###############

# There are multiple entries for a single date. 
# We need to group them into one entry, i.e. one date should have one row ONLY.

# Also, columns of interest are Order Date and Sales. 
# So we will add up all sales for each day.
# Lets Sum up (aggregate) Sales by "Order Date" column
FullRaw.shape
FullRaw2 = FullRaw.groupby('Order Date')['Sales'].sum().reset_index().copy()
FullRaw2["Order Date"].dtypes

# Ideally we should have 1460 rows or days [4years*365days]. But we dont. 
# So everyday's sales data is NOT present. That means we NEED to roll-up further.
# Time Series cannot have NA values.  

###############
# Convert date column to "datetime" variable
###############

# Before further roll-up, lets first convert Order Date column 
# into a "Date" format column and we will set it as the index 
# as well, as that helps in plotting/ conducting time series analysis

FullRaw2['Order Date'] = pd.to_datetime(FullRaw2['Order Date'])
FullRaw2.sort_values("Order Date", inplace = True)
FullRaw2.set_index('Order Date', inplace=True) 

# Setting the index to "datetime" helps in plotting time series

FullRaw2.index.min()
FullRaw2.index.max()
###############
# Further data roll up
###############

# The current data is at a daily level(with lot of "missing days"). 
# In time series analysis, it is usually advisable
# to roll up/ aggregate the data at a higher level, like week or month
# We will consider averaging (rollup) the sales at monthly level 
# (But, you can also try weekly level rollup in the same manner)

FullRaw3 = FullRaw2['Sales'].resample('MS').mean() # MS stands for Month Start. You should get 48 rows (4Years*12Months) 

# You can think of "resample" function as "rollup" function

# Task: Try with weekly data
# FullRaw3 = FullRaw2['Sales'].resample('W').mean() # W stands for Weekly rollup

# Check the table in the below link for more roll up options like 'MS', 'W':
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

###############
# Plot the data
###############

import seaborn as sns
sns.lineplot(data = FullRaw3)

# from matplotlib.pyplot import figure, close
# figure()
# sns.lineplot(x = FullRaw3.index, y = FullRaw3)
# close()

###############
# Time series decomposition
###############

from statsmodels.tsa.seasonal import seasonal_decompose
Decomposed_Series = seasonal_decompose(FullRaw3)
Decomposed_Series.plot()

###############
# Sampling: Train and Test Split
###############

Train = FullRaw3[:36].copy() # First 3 years for training (2014-2016)
Test = FullRaw3[36:].copy() # Last 1 year for testing (2017)

###############
# SES Model
###############

## Model
SES = SimpleExpSmoothing(Train).fit(smoothing_level=0.01) # Model building
SES.summary()
Forecast = SES.forecast(12).rename('Forecast') # Model Forecasting
Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1) 

# Act, Forecast combining

## Plot
import seaborn as sns
sns.lineplot(data = Actual_Forecast_Df)

## Validation
import numpy as np
Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['Sales'] - Validation_Df['Forecast'])**2)) # RMSE

# Creating function

def myFun(model):
    Forecast = model.forecast(12).rename('Forecast') # Model Forecasting
    # Act, Forecast combining
    Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1) 
    # Plot
    sns.lineplot(data = Actual_Forecast_Df)
    ## Validation
    Validation_Df = Actual_Forecast_Df[-12:].copy()
    print("MAPE",np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100) # MAPE
    print("RMSE",np.sqrt(np.mean((Validation_Df['Sales'] - Validation_Df['Forecast'])**2))) # RMSE


###############
# DES Model
###############

## Model
DES = Holt(Train).fit(smoothing_level=0.01, smoothing_slope=0.6)
DES.summary()
Forecast = DES.forecast(12).rename('Forecast')
Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1)

# ## Plot
sns.lineplot(data = Actual_Forecast_Df)

# ## Validation
Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['Sales'] - Validation_Df['Forecast'])**2)) # RMSE


myFun(DES)

###############
# TES Model
###############

## Model
TES = ExponentialSmoothing(Train, 
                           seasonal_periods=12, 
                           seasonal='add',
                           trend = 'add').fit(smoothing_level=0.01, 
                                      smoothing_slope=0.1, 
                                      smoothing_seasonal = 0.3) 
TES.summary()
myFun(TES)
# trend = 'add' means additive trend. Use this when trend is NOT exponentially increasing/decreasing, 
# like a steep increase or decrease 
# seasonal = 'add' means additive seasonality. Use this when seasonality is NOT increasing/decreasing in magnitude
Forecast = TES.forecast(12).rename('Forecast')
Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1)

## Plot
sns.lineplot(data = Actual_Forecast_Df)

## Validation
Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['Sales'] - Validation_Df['Forecast'])**2)) # RMSE

###############
# TES: Automatic selection of level, trend and seasonal smoothing components (alpha, beta, gamma)
###############

## Model
TES2 = ExponentialSmoothing(Train, 
                           seasonal_periods=12, 
                           seasonal='add',
                           trend='add').fit() # Leave fit() empty, then it will automatically find alpha, beta, gamma

myFun(TES2)
Forecast = TES2.forecast(12).rename('Forecast')
Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1)

## Plot
sns.lineplot(data = Actual_Forecast_Df)

## Validation
Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['Sales'] - Validation_Df['Forecast'])**2)) # RMSE


###############
# TES Grid Search
###############

myAlpha = np.round(np.arange(0,1.1,0.1),2)
myBeta = np.round(np.arange(0,1.1,0.1),2)
myGamma = np.round(np.arange(0,1.1,0.1),2)

#list(myAlpha)

alphaList = []
betaList = []
gammaList = []
mapeList = []

for alpha in myAlpha:
    for beta in myBeta:
        for gamma in myGamma:
            
            print(alpha, beta, gamma)
                                              
            TES = ExponentialSmoothing(Train, 
                           seasonal_periods=12, 
                           seasonal='mul', # seasonal='add'
                           trend = 'add').fit(smoothing_level = alpha, 
                                      smoothing_slope = beta, 
                                      smoothing_seasonal = gamma)
            
            Forecast = TES.forecast(12).rename('Forecast')
            Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1)
            Validation_Df = Actual_Forecast_Df[-12:].copy()
            tempMAPE = np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100 # MAPE
            
            
            alphaList.append(alpha)
            betaList.append(beta)
            gammaList.append(gamma)
            mapeList.append(tempMAPE)
            
            
evaluationDf = pd.DataFrame({"alpha": alphaList,
                             "beta": betaList,
                             "gamma": gammaList,
                             "MAPE": mapeList})        
            
evaluationDf.to_csv("EvaluationDf.csv")
###############
###############
# ARIMA
###############
###############


###############
# ARIMA Manual Model
###############

#pip install pmdarima
from pmdarima.arima import ARIMA

## Model
arimaModel = ARIMA((2,1,0), (1,0,0,12)).fit(Train)
Forecast = pd.Series(arimaModel.predict(12)).rename('Forecast')
Forecast
Forecast.index = Test.index # Needed for the pd.concat to work correctly in the next line
Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1)

## Plot
import seaborn as sns
sns.lineplot(data = Actual_Forecast_Df)

## Validation
import numpy as np
Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['Sales'] - Validation_Df['Forecast'])**2)) # RMSE



###############
# ARIMA Automated Model
###############

from pmdarima import auto_arima
arimaModel2 = auto_arima(Train, m = 12)

## Get the order of p,d,q & P,D,Q
arimaModel2.get_params()['order'] # p,d,q
arimaModel2.get_params()['seasonal_order'] # P,D,Q

## Forecasting
Forecast = pd.Series(arimaModel2.predict(12)).rename('Forecast')
Forecast.index = Test.index
Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1)

## Plot
sns.lineplot(data = Actual_Forecast_Df)

## Validation
Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['Sales'] - Validation_Df['Forecast'])**2)) # RMSE

###############
# Adding some parameters (like stationary = True) and changing data distribution using log(Series)
###############

# TrainCopy = Train.copy()
sns.lineplot(data = Train)

trainLog = np.log(Train) # Take log of series
sns.lineplot(data = trainLog)

## Model
# If you believe that the series is stationary, then its worth a shot

arimaModel3 = auto_arima(trainLog, m = 12, stationary = True) 

## Get the order of p,d,q & P,D,Q
arimaModel3.get_params()['order']
arimaModel3.get_params()['seasonal_order']


## Forecasting
Forecast = pd.Series(arimaModel3.predict(12)).rename('Forecast')
Forecast = np.exp(Forecast) # De-log the forecasted values (back to original scale)
Forecast.index = Test.index
Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1)

## Plot
sns.lineplot(data = Actual_Forecast_Df)

## Validation
Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['Sales'] - Validation_Df['Forecast'])**2)) # RMSE




################
# ARIMA Grid Search
################

p = range(0,2)
d = range(0,2)
q = range(0,2)
P = range(2)
D = range(2)
Q = range(2)

pList = []
dList = []
qList = []
PList = []
DList = []
QList = []
mapeList = []
#seasonalPDQList = []

for i in p:
    for j in d:
        for k in q:
            for I in P:
                for J in D:
                    for K in Q:
            
                        print(i,j,k, I, J, K)
                        tempArimaModel = ARIMA((i,j,k), (I,J,K,12)).fit(Train)
                        
                        Forecast = pd.Series(tempArimaModel.predict(12)).rename('Forecast')
                        Forecast.index = Test.index
                        Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1)
                        Validation_Df = Actual_Forecast_Df[-12:].copy()
                        tempMAPE = np.mean(abs(Validation_Df['Sales'] - Validation_Df['Forecast'])/Validation_Df['Sales'])*100 # MAPE
                        
                        pList.append(i)
                        dList.append(j)
                        qList.append(k)
                        PList.append(I)
                        DList.append(J)
                        QList.append(K)
                        mapeList.append(tempMAPE)
            
            
arimaEvaluationDf = pd.DataFrame({"p": pList,
                             "d": dList,
                             "q": qList,
                             "P": PList,
                             "D": DList,
                             "Q": QList,
                             "MAPE": mapeList})        
            
arimaEvaluationDf.to_csv("ArimaEvaluation.csv")
###############
# Finalize arima full model (This is very important)
###############

arimFinalModel = ARIMA((0,1,1), (0,1,0,12)).fit(FullRaw3)

## Forecasting
Forecast = pd.Series(arimFinalModel.predict(12)).rename('Forecast')

# Set the correct dates as index of the forecast obtained in the previous line
start = "2018-01-01" # Check the order/ date format in FullRaw3.index. Its Year-Month-Day.
end = "2018-12-01"
futureDateRange = pd.date_range(start, end, freq='MS')
futureDateRange
Forecast.index =  futureDateRange 

Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1) # Column wise binding

## Plot
sns.lineplot(data = Actual_Forecast_Df)


## Connected line plot
Actual_Forecast_Series = pd.concat([FullRaw3, Forecast], axis = 0) # Row wise binding
sns.lineplot(data = Actual_Forecast_Series)
from matplotlib.pyplot import vlines
vlines(x = Actual_Forecast_Series.index[48], ymin = 0, ymax = max(Actual_Forecast_Series), colors = "red")















# ###############
# # Finalize TES full model (This is very important)
# ###############


TES_Final_Model = ExponentialSmoothing(FullRaw3, 
                            seasonal_periods=12, 
                            seasonal='add',
                            trend = 'add').fit(smoothing_level=0.0662629, 
                                       smoothing_slope=0.0662519, 
                                       smoothing_seasonal = 0.3496301) 


Forecast = TES_Final_Model.forecast(12).rename('Forecast')
Actual_Forecast_Df = pd.concat([FullRaw3, Forecast], axis = 1)

# ## Plot
sns.lineplot(data = Actual_Forecast_Df)

