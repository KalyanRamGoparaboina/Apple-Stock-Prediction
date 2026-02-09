import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Generate sample data (same as in streamlit_app.py)
np.random.seed(42)
time_pts = np.arange(200)
data = 10 + 0.1 * time_pts + 5 * np.sin(2 * np.pi * time_pts / 12) + np.random.normal(0, 1, 200)
data_series = pd.Series(data)

# Create lag features
df = pd.DataFrame(data_series, columns=['val'])
for i in range(1, 4):
    df[f'lag_{i}'] = df['val'].shift(i)
df.dropna(inplace=True)

X = df.drop('val', axis=1)
y = df['val']

# Train XGBoost model
model = XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
model.fit(X, y)

# Save the trained model
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ XGBoost model trained and saved as 'xgboost_model.pkl'")
print(f"Model trained on {len(X)} samples with 3 lag features")
