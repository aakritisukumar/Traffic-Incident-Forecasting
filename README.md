# 📊 Alcohol-Related Accident Forecasting using SARIMA, LSTM, GRU, and Hybrid Models

This project presents a comprehensive time series forecasting system to predict alcohol-related traffic accidents in Germany. It leverages both classical statistical models like SARIMA and advanced deep learning models like LSTM, GRU, and a Hybrid SARIMA+GRU model.

🛠️ Built with Python, Streamlit, and various ML/DL libraries

## 📂 Dataset
The data is sourced from Germany’s official accident statistics:

🔗 Monatszahlen Verkehrsunfälle – GENESIS-Online (Destatis)

## 🚀 Features

- ✅ Time series preprocessing (resampling, filtering, log transform)
- 📉 SARIMA forecasting with seasonal tuning
- 🧠 Deep Learning models:
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
- 🔁 Hybrid model: SARIMA + GRU
- 📊 Streamlit dashboard for model selection and visualization

---

## 📈 Models & Results

### 1. **SARIMA (3,1,3)x(1,0,1,12)**
- Captures trend & seasonality
- Low AIC = **105.66**
- Log-likelihood = **-43.83**
- Limitations: struggles with sudden spikes or irregularities

### 2. **LSTM**
- Struggles to capture spikes
- Underfits the data
- Good for nonlinear sequences, but not optimal here

### 3. **GRU**
- Better than LSTM
- Attempts to follow short-term variations
- Still lags during peak accident months

### 4. **Hybrid SARIMA + GRU**
- Best of both worlds
- SARIMA handles trend & seasonality
- GRU models unpredictable residuals
- **Best overall RMSE and forecast stability**

---
