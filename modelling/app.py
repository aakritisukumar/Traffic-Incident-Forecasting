import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lstm import train_lstm_model, evaluate_lstm_model, plot_lstm_results
from gru import train_gru_model, evaluate_gru_model, plot_gru_results
from hybrid import run_hybrid_model
from plots import plot_monthly_trends, plot_annual_totals, plot_trend_line, plot_seasonal_decomposition, plot_acf_pacf
from preprocessing import load_and_preprocess_data

# Streamlit App
st.set_page_config(layout="wide")
st.title("📊 Alcohol-Related Accidents Forecasting Dashboard")

# Load the preprocessed data
df, monthly_data, annual_data = load_and_preprocess_data()
st.success("Preprocessed data loaded successfully!")

df['WERT_log'] = np.log(df['WERT'].replace(0, np.nan).dropna())

st.subheader("📈 Exploratory Data Visualizations")

# Display monthly trends and annual totals
col1, col2 = st.columns(2)
with col1:
    fig1 = plot_monthly_trends()
    st.pyplot(fig1)

with col2:
    fig2 = plot_annual_totals()
    st.pyplot(fig2)

# Plot trend line for the overall dataset (df)
st.pyplot(plot_trend_line())

# Seasonal Decomposition
st.subheader('📅 Seasonal Decomposition')
fig3 = plot_seasonal_decomposition()
st.pyplot(fig3)

# ACF and PACF Plots
st.subheader('📉 Autocorrelation and Partial Autocorrelation')
fig4 = plot_acf_pacf()
st.pyplot(fig4)

# Select forecasting model
st.subheader("🤖 Select Forecasting Model")
model_choice = st.selectbox("Choose Model", ['LSTM', 'GRU', 'Hybrid SARIMA+GRU'])

if model_choice == 'LSTM':
    with st.spinner("Training LSTM model..."):
        model, scaler, X_test, y_test = train_lstm_model(df)
        actual_exp, pred_exp, rmse = evaluate_lstm_model(model, scaler, X_test, y_test)
    st.success(f"LSTM Model RMSE: {rmse:.4f}")
    st.pyplot(plot_lstm_results(actual_exp, pred_exp))

elif model_choice == 'GRU':
    with st.spinner("Training GRU model..."):
        model, scaler, X_test, y_test = train_gru_model(df)
        actual_exp, pred_exp, rmse = evaluate_gru_model(model, scaler, X_test, y_test)
    st.success(f"GRU Model RMSE: {rmse:.4f}")
    st.pyplot(plot_gru_results(actual_exp, pred_exp))

elif model_choice == 'Hybrid SARIMA+GRU':
    with st.spinner("Running Hybrid SARIMA + GRU model..."):
        model, sarima_result, hybrid_forecast, rmse = run_hybrid_model(df)
    st.success(f"Hybrid Model RMSE: {rmse:.4f}")

    actual = df['WERT_log'][-len(hybrid_forecast):]
    pred_exp = np.exp(hybrid_forecast)
    actual_exp = np.exp(actual)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_exp, label='Actual')
    ax.plot(pred_exp, label='Hybrid Forecast')
    ax.set_title('Hybrid Forecast vs Actual')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Accident Counts')
    ax.legend()
    st.pyplot(fig)
