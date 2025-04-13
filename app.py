#!/usr/bin/env python
# coding: utf-8



# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

st.title("⛅ Vapor Pressure Forecasting App")

# File upload: expects a CSV file with columns "date" and "VPact"
uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV and prepare the time series
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    ts = df['VPact'].dropna()
    
    st.write("### Raw Time Series")
    st.line_chart(ts)
    
    # Time Series Decomposition Section
    st.subheader("Time Series Decomposition")
    decomposition_type = st.selectbox("Decomposition Type", ["Additive", "Multiplicative"])
    period = st.slider("Seasonality Period (days)", 7, 60, 30)
    
    # Check for multiplicative conditions
    if decomposition_type.lower() == "multiplicative" and (ts <= 0).any():
        st.error("Multiplicative decomposition cannot be performed on non-positive values.")
    else:
        decomp = seasonal_decompose(ts, model=decomposition_type.lower(), period=period)
        fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        decomp.observed.plot(ax=ax[0], title='Observed')
        decomp.trend.plot(ax=ax[1], title='Trend')
        decomp.seasonal.plot(ax=ax[2], title='Seasonal')
        decomp.resid.plot(ax=ax[3], title='Residual')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Forecasting Section
    st.subheader("Forecasting")
    model_choice = st.selectbox("Forecasting Model", ["ARIMA", "ETS", "Prophet"])
    forecast_steps = st.slider("Forecast Horizon (days)", 10, 90, 30)
    
    # Split data into training and testing sets
    train = ts[:-forecast_steps]
    test = ts[-forecast_steps:]
    
    if model_choice == "ARIMA":
        model_fit = ARIMA(train, order=(5, 1, 0)).fit()
        forecast = model_fit.forecast(steps=forecast_steps)
    elif model_choice == "ETS":
        model_fit = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=period).fit()
        forecast = model_fit.forecast(steps=forecast_steps)
    else:  # Prophet
        df_prophet = df.reset_index()[['date', 'VPact']].rename(columns={'date': 'ds', 'VPact': 'y'})
        m = Prophet()
        m.fit(df_prophet[:-forecast_steps])
        future = m.make_future_dataframe(periods=forecast_steps)
        forecast_df = m.predict(future)
        forecast = forecast_df['yhat'].iloc[-forecast_steps:].values
        st.write("### Prophet Forecast Plot")
        fig_prophet = m.plot(forecast_df)
        st.pyplot(fig_prophet)
    
    # Evaluation Metrics
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = mean_absolute_percentage_error(test, forecast)
    
    st.write("### Evaluation Metrics")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2%}")
    
    # Plot forecast vs actual
    st.write("### Forecast vs Actual")
    forecast_series = pd.Series(forecast, index=test.index)
    fig_final, ax_final = plt.subplots(figsize=(10, 5))
    train.plot(ax=ax_final, label="Train", color='blue')
    test.plot(ax=ax_final, label="Test", color='orange')
    forecast_series.plot(ax=ax_final, label="Forecast", color="green", linewidth=3, marker="o")
    ax_final.legend()
    ax_final.set_ylabel("Vapor Pressure (hPa)")
    st.pyplot(fig_final)


