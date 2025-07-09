import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Layer, Bidirectional
from keras.callbacks import EarlyStopping
from keras.losses import Huber
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import pickle
import tensorflow as tf
from preprocessing import load_and_preprocess_data  

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

# Train and run hybrid model
def run_hybrid_model(df, seq_len=30, sarima_order=(3, 0, 3), seasonal_order=(2, 1, 2, 12), alpha=0.3,
                     model_path='models/hybrid_model.h5', pickle_path='models/hybrid_model.pkl'):

    # SARIMA model
    sarima_model = SARIMAX(df['WERT_log'], order=sarima_order, seasonal_order=seasonal_order)
    sarima_result = sarima_model.fit(disp=False)
    sarima_pred = sarima_result.fittedvalues
    residuals = df['WERT_log'] - sarima_pred

    # GRU input
    residual_values = residuals.dropna().values
    X, y = [], []
    for i in range(len(residual_values) - seq_len):
        X.append(residual_values[i:i + seq_len])
        y.append(residual_values[i + seq_len])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # GRU model
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=True), input_shape=(seq_len, 1)),
        Dropout(0.2),
        Bidirectional(GRU(32)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=Huber(delta=1.0))
    model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0)

    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    with open(pickle_path, "wb") as f:
        pickle.dump({'model': model, 'sarima_model': sarima_result}, f)
    print(f"Hybrid model saved successfully to: {pickle_path}")

    # Forecasting
    sarima_forecast = sarima_result.get_forecast(steps=len(y)).predicted_mean
    predicted_residuals = model.predict(X, verbose=0)
    hybrid_forecast = alpha * sarima_forecast[-len(predicted_residuals):].values + \
                      (1 - alpha) * (sarima_forecast[-len(predicted_residuals):].values + predicted_residuals.flatten())

    rmse = evaluate_hybrid_model(df['WERT_log'][-len(hybrid_forecast):], hybrid_forecast)
    plot_hybrid_results(df['WERT_log'][-len(hybrid_forecast):], hybrid_forecast)

    return model, sarima_result, hybrid_forecast, rmse

# Evaluation function (for RMSE)
def evaluate_hybrid_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f'Hybrid Model RMSE: {rmse:.4f}')
    return rmse

# Plotting function
def plot_hybrid_results(actual, predicted):
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Hybrid Forecast')
    plt.title('Hybrid SARIMA + GRU Forecast vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Log of WERT')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df, monthly_data, annual_data = load_and_preprocess_data()
    
    if 'WERT_log' not in df.columns:
        if 'WERT' in df.columns:
            df['WERT_log'] = np.log(df['WERT'] + 1)
        else:
            raise ValueError("Dataset must contain 'WERT_log' or 'WERT' column.")
    
    run_hybrid_model(df)
