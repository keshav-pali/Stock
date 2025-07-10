# train_model.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Make model directory
os.makedirs("model", exist_ok=True)

# Download historical stock data
data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")
data.dropna(inplace=True)

# Features: Current day's data, Target: Next day's Close price
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values[:-1]
y = data['Close'].values[1:]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Model trained and saved.")
