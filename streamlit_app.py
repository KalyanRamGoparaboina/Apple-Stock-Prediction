import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import time

# Styling and Page Config
st.set_page_config(
    page_title="ProphetFlow | Time Series Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look (Simplified for Cloud Stability)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #ffffff;
    }
    .stApp {
        background-color: transparent;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    h1, h2, h3 {
        color: #8b5cf6 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Try importing TF for LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    HAS_TF = True
except ImportError:
    HAS_TF = False

# --- Data Generation ---
@st.cache_data
def generate_data(n_points=200):
    np.random.seed(42)
    time_pts = np.arange(n_points)
    # Trend + Seasonality + Noise (Exact formula from time_series_models.py)
    data = 10 + 0.1 * time_pts + 5 * np.sin(2 * np.pi * time_pts / 12) + np.random.normal(0, 1, n_points)
    return pd.Series(data)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/isometric/100/line-chart.png", width=80)
st.sidebar.title("Forecasting Engine")
st.sidebar.markdown("---")

data_points = st.sidebar.slider("Data Points", 50, 500, 200)
forecast_steps = st.sidebar.number_input("Forecast Steps", 1, 50, 10)

selected_models = st.sidebar.multiselect(
    "Select Models",
    ["ARIMA", "SARIMA", "XGBoost", "LSTM"] if HAS_TF else ["ARIMA", "SARIMA", "XGBoost"],
    default=["ARIMA", "SARIMA", "XGBoost"]
)

# --- Main Page ---
st.title("📈 Time Series Forecasting")
st.markdown("### AI-Powered Multi-Model Deployment")

data_series = generate_data(data_points)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Real-time Training & Forecasting")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data_series.index, data_series.values, label='Original Data', color='#475569', linewidth=2, alpha=0.8)
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    ax.spines['bottom'].set_color('#ffffff')
    ax.spines['top'].set_color('#ffffff') 
    ax.spines['right'].set_color('#ffffff')
    ax.spines['left'].set_color('#ffffff')
    ax.tick_params(axis='x', colors='#ffffff')
    ax.tick_params(axis='y', colors='#ffffff')
    ax.grid(color='#1e293b', linestyle='--')

    forecast_index = np.arange(len(data_series), len(data_series) + forecast_steps)
    
    results = {}

    if st.button("Execute Models"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- ARIMA ---
        if "ARIMA" in selected_models:
            status_text.text("Logic: ARIMA(5, 1, 0)...")
            model = ARIMA(data_series, order=(5, 1, 0))
            fit = model.fit()
            forecast = fit.forecast(steps=forecast_steps)
            results['ARIMA'] = forecast
            ax.plot(forecast_index, forecast, label='ARIMA Forecast', color='#3b82f6', linestyle='--')
            progress_bar.progress(25)

        # --- SARIMA ---
        if "SARIMA" in selected_models:
            status_text.text("Logic: SARIMAX(1, 1, 1)x(1, 1, 1, 12)...")
            model = SARIMAX(data_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fit = model.fit(disp=False)
            forecast = fit.forecast(steps=forecast_steps)
            results['SARIMA'] = forecast
            ax.plot(forecast_index, forecast, label='SARIMA Forecast', color='#10b981', linestyle='--')
            progress_bar.progress(50)

        # --- XGBoost ---
        if "XGBoost" in selected_models:
            status_text.text("Logic: XGBRegressor with 3 Lags...")
            df = pd.DataFrame(data_series, columns=['val'])
            for i in range(1, 4):
                df[f'lag_{i}'] = df['val'].shift(i)
            df.dropna(inplace=True)
            X = df.drop('val', axis=1)
            y = df['val']
            model = XGBRegressor(n_estimators=100)
            model.fit(X, y)
            
            # Multi-step
            lags = list(data_series.values[-3:][::-1])
            xgb_forecast = []
            for _ in range(forecast_steps):
                pred = model.predict(np.array([lags]).reshape(1, -1))[0]
                xgb_forecast.append(pred)
                lags = [pred] + lags[:-1]
            
            results['XGBoost'] = xgb_forecast
            ax.plot(forecast_index, xgb_forecast, label='XGBoost Forecast', color='#f59e0b', linestyle='--')
            progress_bar.progress(75)

        # --- LSTM ---
        if "LSTM" in selected_models and HAS_TF:
            status_text.text("Logic: LSTM with 10 Look-back...")
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))
            X_lstm, y_lstm = [], []
            look_back = 10
            for i in range(look_back, len(scaled_data)):
                X_lstm.append(scaled_data[i-look_back:i, 0])
                y_lstm.append(scaled_data[i, 0])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
            
            lstm_model = Sequential([
                LSTM(50, activation='relu', input_shape=(look_back, 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_lstm, y_lstm, epochs=10, verbose=0)
            
            # Forecast
            curr_batch = scaled_data[-look_back:].reshape((1, look_back, 1))
            lstm_forecast = []
            for _ in range(forecast_steps):
                pred_scaled = lstm_model.predict(curr_batch, verbose=0)
                lstm_forecast.append(pred_scaled[0][0])
                curr_batch = np.append(curr_batch[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)
            
            final_lstm = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()
            results['LSTM'] = final_lstm
            ax.plot(forecast_index, final_lstm, label='LSTM Forecast', color='#ec4899', linestyle='--')
            progress_bar.progress(100)

        status_text.success("Deployment Sync Complete!")
        ax.legend(facecolor='#1e293b', edgecolor='#ffffff', labelcolor='#ffffff')
        st.pyplot(fig)

with col2:
    st.markdown("#### Intelligence Metrics")
    if not results:
        st.info("Start engine to view metrics.")
    else:
        for m_name, m_data in results.items():
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 5px;">{m_name} Engine</p>
                <h3 style="margin: 0; color: #ffffff;">Operational</h3>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

# --- Data Table ---
with st.expander("Raw Data Source"):
    st.dataframe(data_series, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>AI Forecasting Dashboard | Built from Original Source Code</p>", unsafe_allow_html=True)
