import streamlit as st
import pandas as pd
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import starfishX as ax
from datetime import date
import plotly.graph_objects as go


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator 

START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


st.title('Time Series app 📈')
stocks = ("BANPU","ADVANC","AOT","CPALL")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)
#n_years = st.slider("Years of prediction:", 1, 4) 
#period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = ax.loadHistData_v2 (ticker, START, TODAY )
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).data.date
    data.reset_index(inplace=True)
    return(data)

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Load data...done! ")

st.subheader('Stock data: {}📶'.format(selected_stocks))
st.write(data)

st.subheader('Statistic of stock data 📊 ')
st.write(data.describe())



def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_close'))
    fig.layout.update(title = "Time Series Data")
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(count=6,
                        label="6m",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="YTD",
                        step="year",
                        stepmode="todate"),
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    st.plotly_chart(fig)
plot_raw_data()

# Technical Indicators
def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose Technical Indicator ', ['BB', 'MACD', 'RSI', 'SMA', 'EMA'])
    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()
    
    if option == 'BB':
        st.write('BB')
        st.line_chart(bb)
        
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)
tech_indicators()

# Forecasting
data = data[['Date', 'Close']]
data.set_index('Date', inplace=True)
st.subheader('Data ก่อนนำไปใช้กับ ARIMA')
st.write(data)

 # เช็ค stationaty
def stationary_test(data):
    res = testing(data)
    st.text(f'Augmented Dickey_fuller Statistical Test: {res[0]} \
           \np-values: {res[1]}')
    st.text('Critical values at different levels:')
    for k, v in res[4].items():
        st.text(f'{k}:{v}')
    if res[1] > 0.05:
        st.text('Your data is non-stationary and is being transformed \
               \nto a stationary time series data. ')
        if st.button('Check results'):
            data_transform(data)
    elif res[1] <= 0.05:
        st.text('Your data is stationary and is ready for training.')

def testing(df):
    return adfuller(df)
   
def data_transform(df):
    df_log = np.log(df.iloc[:, 0])
    df_diff = df_log.diff().dropna()
    res = testing(df)
    if res[1] < 0.05:
        st.line_chart(df_diff)
        st.write('1st order differencing')
    else:
        df_diff_2 = df_diff.diff().dropna()
        st.line_chart(df_diff_2)
        st.write('2nd order differencing')
        stationary_test(df_diff_2)
 

st.subheader('ตรวจสอบค่าความคงที่ของข้อมูล (stationary) 📄')
stationary_test(data)


# หา p,q,d ที่ดีที่สุดมีค่า rmse ต่ำที่สุดโดยใช้ grid-search
def optimum_para(df):
    p_values = [0, 1, 2]
    d_values = [0, 1, 2]
    q_values = [0, 1, 2]
    size = int(len(df) * .7)
    train, test = df[:size], df[size:]
    for p in p_values:
        for q in q_values:
            for d in d_values:
                order = (p,q,d)
                model = sm.tsa.arima.ARIMA(train, order=order).fit()
                preds = model.predict(start=len(train), end=len(train) + len(test)-1)
                error = sqrt(mean_squared_error(test, preds))
                st.text(f'ARIMA {order} RMSE: {error}')

# forecat
def forecast_data(df):
    if st.button('Find Optimun Parameter'):
        st.text('...ค้นหา optimum parameter')
        optimum_para(df)
    st.text('Enter the parameter with the lowest RMSE')
    p = st.number_input('The p term', min_value=0, max_value=2, step= 1)
    q = st.number_input('The q term', min_value=0, max_value=2, step= 1)
    d = st.number_input('The d term', min_value=0, max_value=2, step= 1)
    period = st.number_input('Enter the next period(s) you want to forecast', value=7)
    button = st.button('Forecast')
    if button:
        model_forecast(df, p, q, d, period)


def model_forecast(data, p, q, d, period):
    size = int(len(data) * .7)
    train, test = data[:size], data[size:]
    model = sm.tsa.arima.ARIMA(train.values, order=(p,q,d))
    model_fit = model.fit()
    output = model_fit.predict(start=len(train), end = len(train)+ len(test)-1)
    error = sqrt(mean_squared_error(output, test))
    st.text(f'RMSE using {p,q,d}: {error}')
    st.text(f'Forecasting {period} future values')
    model_2 = sm.tsa.arima.ARIMA(data.values, order = (p,q,d)).fit()
    forecast = model_2.predict(start=len(data), end=len(data) + period, typ='levels')
    day = 1
    for i in forecast:
        st.text(f'Period {day}: {i}')
        day += 1
forecast_data(data)

