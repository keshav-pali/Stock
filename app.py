# app.py
from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    predictionval = None
    if request.method == "POST":
        symbol = request.form["symbol"]
        df = yf.download(symbol, period="10d")
        if len(df) < 2:
            print("Inside if")
            predictionval = "Not enough data!"
        else:
            print("Inside else")
            latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-2].values.reshape(1, -1)
            print("Latest: ", latest)
            scaled = scaler.transform(latest)
            print("Scaled: ", scaled)
            predicted = model.predict(scaled)[0]
            print("Predicted: ", predicted)
            predictionval = f"{predicted[0]:.2f}"
            print("Predictionval: ", predictionval)
    return render_template("index.html", prediction=predictionval)

if __name__ == "__main__":
    app.run(debug=True)
