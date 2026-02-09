import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Note: Keeping imports here for readability, but make sure they are installed
# For LSTM, we use tensorflow
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    HAS_TF = True
except ImportError:
    HAS_TF = False

# 1. Generate Synthetic Data
def generate_data():
    np.random.seed(42)
    time = np.arange(200)
    # Trend + Seasonality + Noise
    data = 10 + 0.1 * time + 5 * np.sin(2 * np.pi * time / 12) + np.random.normal(0, 1, 200)
    return pd.Series(data)

# 2. ARIMA Model
def build_arima(series):
    print("\n--- Training ARIMA ---")
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    print("ARIMA Forecast (next 10 steps):")
    print(forecast.values)
    return forecast

# 3. SARIMA Model
def build_sarima(series):
    print("\n--- Training SARIMA ---")
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=10)
    print("SARIMA Forecast (next 10 steps):")
    print(forecast.values)
    return forecast

# 4. XGBoost Model (requires lag features)
def build_xgboost(series):
    print("\n--- Training XGBoost ---")
    df = pd.DataFrame(series, columns=['val'])
    # Create lag features
    for i in range(1, 4):
        df[f'lag_{i}'] = df['val'].shift(i)
    df.dropna(inplace=True)
    
    X = df.drop('val', axis=1)
    y = df['val']
    
    # Simple split
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"XGBoost Test Predictions (first 5): {preds[:5]}")
    return model

# 5. LSTM Model
def build_lstm(series):
    if not HAS_TF:
        print("\n--- LSTM skipped (TensorFlow not found) ---")
        return
    
    print("\n--- Training LSTM ---")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, y = [], []
    look_back = 10
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0) # Fewer epochs for speed
    
    # Predict next step
    last_batch = scaled_data[-look_back:].reshape((1, look_back, 1))
    pred_scaled = model.predict(last_batch)
    pred = scaler.inverse_transform(pred_scaled)
    print(f"LSTM Next Step Prediction: {pred[0][0]}")
    return model

if __name__ == "__main__":
    data_series = generate_data()
    
    # Execute each model
    arima_forecast = build_arima(data_series)
    sarima_forecast = build_sarima(data_series)
    xgb_predictions = build_xgboost(data_series)
    # Note: LSTM only returns a single next-step prediction in this basic demo
    
    # Simple Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(data_series.index, data_series.values, label='Original Data', color='gray', alpha=0.5)
    
    # Plot ARIMA/SARIMA forecasts
    forecast_index = np.arange(len(data_series), len(data_series) + 10)
    plt.plot(forecast_index, arima_forecast, label='ARIMA Forecast', color='blue', linestyle='--')
    plt.plot(forecast_index, sarima_forecast, label='SARIMA Forecast', color='green', linestyle='--')
    
    plt.title('Time Series Forecasting Comparison')
    plt.legend()
    plt.savefig('C:/Users/gopar/.gemini/antigravity/scratch/time_series_modeling/forecast_plot.png')
    print("\nVisualization saved as 'forecast_plot.png'")
    
    build_lstm(data_series)
    
    print("\nAll models built successfully!")
