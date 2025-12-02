import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# 1) 데이터 로드
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", auto_adjust=True)
price = df["Close"].reset_index()
price.columns = ["ds", "y"]


# 2) Prophet 기반 30일 미래 예측
prophet_model = Prophet()
prophet_model.fit(price)

future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)

prophet_future = forecast[["ds", "yhat"]].tail(30)
prophet_future = prophet_future.set_index("ds")["yhat"]


# 3) LSTM 기반 30일 미래 예측

# 시계열 스케일링
scaler = MinMaxScaler()
scaled = scaler.fit_transform(price["y"].values.reshape(-1,1))

window = 60
X_lstm, y_lstm = [], []

for i in range(len(scaled) - window - 1):
    X_lstm.append(scaled[i:i+window])
    y_lstm.append(scaled[i+window])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(window,1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.fit(X_lstm, y_lstm, epochs=7, batch_size=32, verbose=0)

# 다중스텝 미래 예측
future_pred = []
last_seq = scaled[-window:].reshape(1, window, 1)

for _ in range(30):
    next_scaled = model.predict(last_seq, verbose=0)[0][0]
    next_price = scaler.inverse_transform([[next_scaled]])[0][0]
    future_pred.append(next_price)

    last_seq = np.append(last_seq[:,1:,:], [[[next_scaled]]], axis=1)

# LSTM 미래 시리즈
lstm_future_index = pd.date_range(start=price["ds"].iloc[-1], periods=31, freq="D")[1:]
lstm_future = pd.Series(future_pred, index=lstm_future_index)


# 4) 그래프 출력 — 미래만 표시
plt.figure(figsize=(12,6))

plt.plot(prophet_future.index, prophet_future.values, label="Prophet 예측", color="green")
plt.plot(lstm_future.index, lstm_future.values, label="LSTM 예측", color="blue")

plt.title(f"{ticker} 미래 30일 전망 (LSTM + Prophet)")
plt.xlabel("날짜")
plt.ylabel("예측 가격")
plt.legend()
plt.grid(True)
plt.show()