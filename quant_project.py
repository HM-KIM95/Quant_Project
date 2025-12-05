import torch
torch.set_default_dtype(torch.float32)
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from xgboost import XGBRegressor
plt.rcParams['axes.unicode_minus'] = False
rc('font', family='AppleGothic')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from darts import TimeSeries
from darts.models import TFTModel


# =========================
# 1) 데이터 로드
# =========================
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", auto_adjust=True)

df = df[["Close"]].rename(columns={"Close": "y"})


# =========================
# 2) 미래 수익률 라벨 생성 (XGBoost용)
# =========================
df["future_30"] = df["y"].shift(-30) / df["y"] - 1
df = df.dropna()


# =========================
# 3) XGBoost 입력 특징 생성
# =========================
df["ret_1d"] = df["y"].pct_change()
df["ret_5d"] = df["y"].pct_change(5)
df["ret_20d"] = df["y"].pct_change(20)
df["vol_20d"] = df["y"].pct_change().rolling(20).std()
df = df.dropna()

features = ["ret_1d", "ret_5d", "ret_20d", "vol_20d"]
X = df[features]
y = df["future_30"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, shuffle=False
)

# =========================
# 4) XGBoost 학습
# =========================
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
)
xgb_model.fit(X_train, y_train)

# =========================
# 5) XGBoost 30일 미래 수익률 예측
# =========================
last_features = df[features].iloc[-1:].values
last_features_scaled = scaler.transform(last_features)
xgb_pred = xgb_model.predict(last_features_scaled)[0]

print("📌 XGBoost 예측 미래 30일 수익률:", round(xgb_pred * 100, 2), "%")


# =========================
# 6) TFT 기반 미래 가격 30일 예측
# =========================
series = TimeSeries.from_dataframe(
    df,
    value_cols="y",
    fill_missing_dates=True,
    freq="B"
)

tft = TFTModel(
    input_chunk_length=60,
    output_chunk_length=30,
    hidden_size=32,
    lstm_layers=2,
    dropout=0.1,
    batch_size=32,
    n_epochs=30,
    add_relative_index=True,
    pl_trainer_kwargs={"accelerator": "cpu"}
)

tft.fit(series)

tft_future = tft.predict(30)


# =========================
# 7) 앙상블 최종 예측
# =========================
future_curve = tft_future.values().flatten()
future_dates = tft_future.time_index

# XGBoost 기반 단일 미래 30일 가격 예측
last_price = df["y"].iloc[-1]
xgb_pred_price = last_price * (1 + xgb_pred)

ensemble_price = (future_curve[-1] * 0.6) + (xgb_pred_price * 0.4)

print("\n📌 최종 앙상블 예측 30일 뒤 가격:", round(float(ensemble_price), 2))


# =========================
# 8) 그래프 출력
# =========================
plt.figure(figsize=(12,6))

plt.plot(df.index[-60:], df["y"].iloc[-60:], label="실제 종가 (최근 60일)", color="blue")
plt.plot(future_dates, future_curve, label="TFT 미래 가격", color="green")
plt.scatter(future_dates[-1], ensemble_price, color="red", label="앙상블 최종 예측")

plt.title("AAPL 30일 미래 예측 (XGBoost + TFT 앙상블)")
plt.xlabel("날짜")
plt.ylabel("예측 가격")
plt.grid(True)
plt.legend()
plt.show()