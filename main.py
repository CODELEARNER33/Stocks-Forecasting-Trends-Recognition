import streamlit as st
from datetime import date
import requests
import pandas as pd
from neuralprophet import NeuralProphet
from plotly import graph_objs as go

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
API_KEY = "Your API key"

st.title("Stock Prediction App")

# Selecting stock and prediction period
stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction:", stocks)
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "historical" in data:
            df = pd.DataFrame(data["historical"])
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= START]
            df = df.rename(columns={
                "date": "Date",
                "close": "Close",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "volume": "Volume"
            })
            return df
    return pd.DataFrame()  # if data not found return an empty DataFrame 

# Loading and displaying raw data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader("Raw data")
st.write(data.tail())

# Ploting raw data
def plot_raw_data():

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
if not data.empty:
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

    # Training NeuralProphet model
    m = NeuralProphet()
    m.fit(df_train)
    future = m.make_future_dataframe(df_train, periods=period, n_historic_predictions=True)
    forecast = m.predict(future)

    # Displaying forecast data
    st.subheader("Forecast data")
    st.write(forecast.tail())

    # Ploting forecast data
    st.write("Forecast vs Actual Data")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], name='Prediction'))
    fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name='Actual'))
    fig1.layout.update(title_text="Forecast vs Actual", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    # Ploting forecast components
    st.write("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)


else:
    st.error("Failed to load data. Please check the API key or ticker symbol.")
