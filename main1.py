import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf # https://pypi.org/project/yfinance/
from ta.volatility import BollingerBands
#from ta.trend import MACD
#from ta.momentum import RSIIndicator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
import pylab as pl
from sklearn.metrics import mean_squared_error, r2_score
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
import os
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from io import BytesIO
import math
###########
# sidebar #
###########
st.set_page_config(layout="wide",)
option = st.sidebar.selectbox('Select one symbol', ( 'LUV', 'NVDA',"AMZN",'WMT'))
import datetime
today = datetime.date.today()
before = today - datetime.timedelta(days=700)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

nolags = st.sidebar.selectbox('Select the lags you want', ( 5,1,2,3,4,6,7,8,9,10))

##############
# Stock data #
##############

# Download data
df = yf.download(option,start= start_date,end= end_date, progress=False)

df_stock=df.drop(['Open','High','Low','Adj Close','Volume'],axis=1)

###################
# Set up main app #
###################
if option == 'LUV':
    st.markdown("<h1 style='text-align: center; color: black;'>Southwest Airlines Stock Closing Prices over time</h1>", unsafe_allow_html=True)
elif option == 'NVDA':
    st.markdown("<h1 style='text-align: center; color: black;'>NVDIA Stock Closing Prices over time</h1>", unsafe_allow_html=True)
elif option == 'AMZN':
    st.markdown("<h1 style='text-align: center; color: black;'>Amazon Stock Closing Prices over time</h1>", unsafe_allow_html=True)
elif option == 'WMT':
    st.markdown("<h1 style='text-align: center; color: black;'>Walmart Stock Closing Prices over time</h1>", unsafe_allow_html=True)
#st.write('Stock Closing Prices over time')
st.line_chart(df_stock)

progress_bar = st.progress(0)

# Plot MACD
##st.write('Stock Moving Average Convergence Divergence (MACD)')
##st.area_chart(macd)

# Plot RSI
##st.write('Stock RSI ')
##st.line_chart(rsi)

# Data of recent days
st.markdown("<h1 style='text-align: center; color: black;'>Recent Data of the selected Stock</h1>", unsafe_allow_html=True)
st.dataframe(df.tail(10))



avg_stock = df_stock['Close'].mean()
if option == 'LUV':
    st.markdown(f"<h3 style='text-align: left; color: black;'>Average Stock Price of Southwest Airlines is {avg_stock:.2f}</h1>", unsafe_allow_html=True)
elif option == 'NVDA':
    st.markdown(f"<h3 style='text-align: left; color: black;'>Average Stock Price of NVDIA Airlines is {avg_stock:.2f}</h1>", unsafe_allow_html=True)
elif option == 'AMZN':
    st.markdown(f"<h3 style='text-align: left; color: black;'>Average Stock Price of Amazon is {avg_stock:.2f}</h1>", unsafe_allow_html=True)
elif option == 'WMT':
    st.markdown(f"<h3 style='text-align: left; color: black;'>Average Stock Price of Walmart is {avg_stock:.2f}</h1>", unsafe_allow_html=True)


def autocorr(timeseries, nolags):
    fig1 = plt.figure()
    lag_plot(df_stock['Close'], lag=nolags)
    plt.title('Autocorrelation plot with lag = {}'.format(nolags))
    ##plt.title('Southwest Stock - Autocorrelation plot with lag = ', nolags)
    plt.show()
    st.pyplot(fig1)
st.markdown("<h1 style='text-align: center; color: black;'>Autocorrelation plot with selected number of lags</h1>", unsafe_allow_html=True)
autocorr(df_stock, nolags)
    

#Test for staionarity
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    fig = plt.figure()
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    st.markdown("<h1 style='text-align: center; color: black;'>Rolling Mean and Standard Deviation</h1>", unsafe_allow_html=True)
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    st.pyplot(fig)
    center_css = """
    <style>
        .dataframe th, .dataframe td {
            text-align: center !important;
        }
    </style>
    """

    # Inject CSS with Markdown
    st.markdown(center_css, unsafe_allow_html=True)

    # Display DataFrame
    st.markdown("<h1 style='text-align: center; color: black;'>Statistic values of Dickey Filler Test</h1>", unsafe_allow_html=True)
    st.dataframe(output)

    #st.markdown("<h1 style='text-align: center; color: black;'>Statistic values of Dickey Filler Test</h1>", unsafe_allow_html=True)
    #st.write('Statistic values of Dickey Filler Test')
    
    #st.dataframe(output)

    # ... [rest of your code]


    ##st.dataframe(output)
    return output

##st.write('Results of Dickey Filler Test')
test_stationarity(df_stock)

