# app.py
from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    predictionval = None
    high = low = volume = avg = None

    if request.method == "POST":
        symbol = request.form["symbol"]
        df = yf.download(symbol, period="10d")

        if len(df) < 2:
            print("Inside if")
            predictionval = "Not enough data!"
        else:
            print("Inside else")
            # For prediction
            latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-2].values.reshape(1, -1)
            print("Latest: ", latest)
            scaled = scaler.transform(latest)
            print("Scaled: ", scaled)
            predicted = model.predict(scaled)
            print("Predicted: ", predicted)
            predictionval = f"{predicted.item():.2f}"
            print("Predictionval: ", predictionval)

            # Get latest actual day's values
            latest_row = df.iloc[-1]
            high = round(float(latest_row['High']), 2)
            low = round(float(latest_row['Low']), 2)
            volume = int(latest_row['Volume'])
            avg = round((float(latest_row['High']) + float(latest_row['Low'])) / 2, 2)

    return render_template("index.html", prediction=predictionval, high=high, low=low, volume=volume, avg=avg)

if __name__ == "__main__":
    app.run(debug=True)
