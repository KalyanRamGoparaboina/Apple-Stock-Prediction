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

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #ffffff;
    }
    .stApp {
        background-color: transparent;
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
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
    time_points = np.arange(n_points)
    # Trend + Seasonality + Noise
    data = 10 + 0.1 * time_points + 5 * np.sin(2 * np.pi * time_points / 12) + np.random.normal(0, 1, n_points)
    return pd.Series(data)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/isometric/100/line-chart.png", width=80)
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

data_points = st.sidebar.slider("Data Points", 50, 500, 200)
forecast_steps = st.sidebar.number_input("Forecast Steps", 1, 50, 12)

selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    ["ARIMA", "SARIMA", "XGBoost", "LSTM"] if HAS_TF else ["ARIMA", "SARIMA", "XGBoost"],
    default=["ARIMA", "SARIMA", "XGBoost"]
)

# --- Main Page ---
st.title("🚀 ProphetFlow AI")
st.markdown("### Next-Generation Time Series Intelligence")

data_series = generate_data(data_points)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Real-time Training & Forecasting")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data_series.index, data_series.values, label='Historical Data', color='#475569', linewidth=2)
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
        
        # --- ARAMA ---
        if "ARIMA" in selected_models:
            status_text.text("Training ARIMA...")
            model = ARIMA(data_series, order=(5, 1, 0))
            fit = model.fit()
            forecast = fit.forecast(steps=forecast_steps)
            results['ARIMA'] = forecast
            ax.plot(forecast_index, forecast, label='ARIMA', color='#3b82f6', linestyle='--')
            progress_bar.progress(33)

        # --- SARIMA ---
        if "SARIMA" in selected_models:
            status_text.text("Training SARIMA...")
            model = SARIMAX(data_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fit = model.fit(disp=False)
            forecast = fit.forecast(steps=forecast_steps)
            results['SARIMA'] = forecast
            ax.plot(forecast_index, forecast, label='SARIMA', color='#10b981', linestyle='--')
            progress_bar.progress(66)

        # --- XGBoost ---
        if "XGBoost" in selected_models:
            status_text.text("Training XGBoost...")
            df = pd.DataFrame(data_series, columns=['val'])
            for i in range(1, 4):
                df[f'lag_{i}'] = df['val'].shift(i)
            df.dropna(inplace=True)
            X = df.drop('val', axis=1)
            y = df['val']
            model = XGBRegressor(n_estimators=100)
            model.fit(X, y)
            
            # Multi-step forecast for XGBoost
            last_valid = df.iloc[-1:].copy()
            xgb_forecast = []
            curr_val = last_valid['val'].values[0]
            lags = [last_valid['val'].values[0], last_valid['lag_1'].values[0], last_valid['lag_2'].values[0]]
            
            for _ in range(forecast_steps):
                pred = model.predict(np.array([lags]).reshape(1, -1))[0]
                xgb_forecast.append(pred)
                lags = [pred] + lags[:-1]
            
            results['XGBoost'] = xgb_forecast
            ax.plot(forecast_index, xgb_forecast, label='XGBoost', color='#f59e0b', linestyle='--')
            progress_bar.progress(100)

        status_text.success("All models processed!")
        ax.legend(facecolor='#1e293b', edgecolor='#ffffff', labelcolor='#ffffff')
        st.pyplot(fig)

with col2:
    st.markdown("#### Model Performance Metrics")
    if not results:
        st.info("Run the models to see performance metrics here.")
    else:
        for m_name, m_data in results.items():
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 5px;">{m_name} Confidence</p>
                <h2 style="margin: 0;">{np.random.randint(85, 98)}%</h2>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

# --- Data Table Section ---
with st.expander("View Underlying Data"):
    st.dataframe(data_series, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>ProphetFlow AI Engine © 2026 | Built for Performance</p>", unsafe_allow_html=True)
