import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf # https://pypi.org/project/yfinance/
import seaborn
from scipy import stats
import pylab as pl
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


st.set_page_config(layout="wide",)

st.sidebar.markdown("Select the Stock you want to Analyze below:")
st.sidebar.markdown("LUV is for Southwest Airlines")
st.sidebar.markdown("NVDA is for NVIDIA")
st.sidebar.markdown("AMZN is for Amazon")
st.sidebar.markdown("WMT is for Walmart")

option = st.sidebar.selectbox('Select one stock', ( 'LUV', 'NVDA',"AMZN",'WMT'))
import datetime
today = datetime.date.today()
before = today - datetime.timedelta(days=700)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

number = st.sidebar.number_input('Enter the number of days of recent data you want', min_value=1, value=10)
window = st.sidebar.number_input('Select the rolling window for volatality of Stock', min_value=2, value=12)
#window = st.sidebar.selectbox('', (30,10,12,20,40,50,60,70,80,90,100))
nolags = st.sidebar.selectbox('Select the lags you want', (5,1,2,3,4,6,7,8,9,10))



df = yf.download(option,start= start_date,end= end_date, progress=False)

df_stock=df.drop(['Open','High','Low','Adj Close','Volume'],axis=1)

if option == 'LUV':
    st.markdown("<h1 style='text-align: center; color: black;'>Southwest Airlines Stock Closing Prices over time</h1>", unsafe_allow_html=True)
elif option == 'NVDA':
    st.markdown("<h1 style='text-align: center; color: black;'>NVDIA Stock Closing Prices over time</h1>", unsafe_allow_html=True)
elif option == 'AMZN':
    st.markdown("<h1 style='text-align: center; color: black;'>Amazon Stock Closing Prices over time</h1>", unsafe_allow_html=True)
elif option == 'WMT':
    st.markdown("<h1 style='text-align: center; color: black;'>Walmart Stock Closing Prices over time</h1>", unsafe_allow_html=True)
st.line_chart(df_stock)

progress_bar = st.progress(0)




st.markdown("<h1 style='text-align: center; color: black;'>Recent Data of the selected Stock</h1>", unsafe_allow_html=True)

def recent_data(number):
    if isinstance(number, int) and number > 0:
        st.dataframe(df.tail(number))
    else:
        st.error("Please enter a valid positive integer.")

recent_data(number)


if option == 'LUV':
    st.markdown("<h1 style='text-align: center; color: black;'>Southwest Airlines Daily Returns</h1>", unsafe_allow_html=True)
elif option == 'NVDA':
    st.markdown("<h1 style='text-align: center; color: black;'>NVDIA Daily Returns</h1>", unsafe_allow_html=True)
elif option == 'AMZN':
    st.markdown("<h1 style='text-align: center; color: black;'>Amazon Daily Returns</h1>", unsafe_allow_html=True)
elif option == 'WMT':
    st.markdown("<h1 style='text-align: center; color: black;'>Walmart Daily Returns</h1>", unsafe_allow_html=True)
##st.line_chart(df_stock)


df['daily_returns']=(df['Close'].pct_change())*100
fig_daily,ax=plt.subplots(figsize=(12,6))
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.plot(df['daily_returns'], label = 'Daily Returns')
plt.legend(loc='best')
st.pyplot(fig_daily)

avg_stock = df_stock['Close'].mean()
if option == 'LUV':
    st.markdown(f"<h3 style='text-align: left; color: black;'>Average Stock Price of Southwest Airlines is {avg_stock:.2f}</h3>", unsafe_allow_html=True)
elif option == 'NVDA':
    st.markdown(f"<h3 style='text-align: left; color: black;'>Average Stock Price of NVDIA Airlines is {avg_stock:.2f}</h3>", unsafe_allow_html=True)
elif option == 'AMZN':
    st.markdown(f"<h3 style='text-align: left; color: black;'>Average Stock Price of Amazon is {avg_stock:.2f}</h3>", unsafe_allow_html=True)
elif option == 'WMT':
    st.markdown(f"<h3 style='text-align: left; color: black;'>Average Stock Price of Walmart is {avg_stock:.2f}</h3>", unsafe_allow_html=True)


def plot_stock_volatility(stock_df, window):
    daily_returns = stock_df['Close'].pct_change()
    
    rolling_std = daily_returns.rolling(window=window).std()
    
    plt.figure(figsize=(12, 6))
    rolling_std.plot()
    plt.title(f'Volatility of {option} Stock (Rolling {window}-Day Std Dev)')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Std Dev)')
    plt.grid(True)
    st.pyplot(plt)

plot_stock_volatility(df_stock, window)

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
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
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

    
   
    st.markdown("<h1 style='text-align: center; color: black;'>Statistic values of Dickey Filler Test</h1>", unsafe_allow_html=True)
    st.dataframe(output)
    return output

test_stationarity(df_stock)

