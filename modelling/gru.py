import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import joblib
import pickle
from preprocessing import load_and_preprocess_data  

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def evaluate_gru_model(model, scaler, X_test, y_test):
    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    pred_exp = np.exp(pred)
    actual_exp = np.exp(actual)

    rmse = np.sqrt(mean_squared_error(actual_exp, pred_exp))
    return actual_exp, pred_exp, rmse

def plot_gru_results(actual_exp, pred_exp):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_exp, label='Actual')
    ax.plot(pred_exp, label='Predicted')
    ax.set_title('GRU Forecast vs Actual')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Accident Counts')
    ax.legend()
    return fig

def train_gru_model(df, column='WERT_log', window_size=12,
                    model_path='models/gru_model.h5',
                    scaler_path='models/scaler.gz',
                    pickle_path='models/gru_model.pkl'):

    data = df[column].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, window_size)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    with open(pickle_path, "wb") as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"Model saved successfully to: {pickle_path}")

    actual_exp, pred_exp, rmse = evaluate_gru_model(model, scaler, X_test, y_test)
    print(f'GRU RMSE: {rmse:.4f}')
    plot_gru_results(actual_exp, pred_exp)
    plt.show()

    return model, scaler, X_test, y_test

if __name__ == "__main__":
    df, monthly_data, annual_data = load_and_preprocess_data()
    if 'WERT_log' not in df.columns:
        df['WERT_log'] = np.log(df['WERT'] + 1)

    train_gru_model(df)
