import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='AppleGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

import requests
from bs4 import BeautifulSoup

# ========= 1. 분석 종목 =============
tickers = [
    "005930.KS", "005380.KS", "058610.KS", "278470.KS",
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META",
    "QQQ", "SCHD", "JEPQ"
]

# ========= 2. 가격 데이터 ============
price = yf.download(tickers, start="2015-01-01", auto_adjust=True)["Close"]

# ========= 3. 가치지표 불러오기 =========
def get_fundamental(ticker):
    if ticker.endswith(".KS"):
        code = ticker.replace(".KS", "")
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")

        # PER
        try:
            per_text = soup.select_one("#_per").text.strip()
            per = float(per_text.replace(",", ""))
        except:
            per = 9999

        # PBR
        try:
            pbr_text = soup.select_one("#_pbr").text.strip()
            pbr = float(pbr_text.replace(",", ""))
        except:
            pbr = 9999

    else:
        info = yf.Ticker(ticker).info
        per = info.get("trailingPE", 9999)
        pbr = info.get("priceToBook", 9999)

    return per, pbr

per, pbr = {}, {}
for t in tickers:
    per_val, pbr_val = get_fundamental(t)
    per[t] = per_val if per_val is not None else 9999
    pbr[t] = pbr_val if pbr_val is not None else 9999

df = pd.DataFrame({
    "Ticker": list(per.keys()),
    "PER": list(per.values()),
    "PBR": list(pbr.values())
}).set_index("Ticker")

# ========= 4. 모멘텀 팩터 ============
momentum_3m = price.iloc[-1] / price.iloc[-63] - 1
momentum_6m = price.iloc[-1] / price.iloc[-126] - 1
momentum_12m = price.iloc[-1] / price.iloc[-252] - 1

df["MOM3"] = momentum_3m
df["MOM6"] = momentum_6m
df["MOM12"] = momentum_12m

# ========= 5. 변동성 팩터 ============
vol = price.pct_change().std()
df["VOL"] = vol

# ========= 5-1. ML 기반 미래 예측 (30일 미래 수익률 예측) ============
# 과거 30일 미래 수익률 계산
future_return = price.shift(-30) / price - 1
future_ret_last = future_return.iloc[:-30]  # 마지막 30일 제거

# ML 학습 데이터셋 구성
factor_df = pd.DataFrame({
    "PER": df["PER"],
    "PBR": df["PBR"],
    "MOM3": df["MOM3"],
    "MOM6": df["MOM6"],
    "MOM12": df["MOM12"],
    "VOL": df["VOL"],
})

# 예측 목표값 생성: 각 종목의 30일 미래 수익률
y = future_ret_last.iloc[-1]

# 결측치 제거
X = factor_df.fillna(0)
y = y.fillna(0)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# 현재 시점에서 30일 미래 수익률 예측
df["PRED_RET_30D"] = model.predict(X)

# ========= 6. 팩터 표준화(랭킹화) ============
df["PER_rank"] = df["PER"].rank()
df["PBR_rank"] = df["PBR"].rank()
df["MOM3_rank"] = df["MOM3"].rank(ascending=False)
df["MOM6_rank"] = df["MOM6"].rank(ascending=False)
df["MOM12_rank"] = df["MOM12"].rank(ascending=False)
df["VOL_rank"] = df["VOL"].rank(ascending=True)

# ========= 7. 종합 스코어 ============
df["PRED_rank"] = df["PRED_RET_30D"].rank(ascending=False)

df["score"] = (
      df["PER_rank"] * 1
    + df["PBR_rank"] * 1
    + df["MOM3_rank"] * 2
    + df["MOM6_rank"] * 2
    + df["MOM12_rank"] * 3
    + df["VOL_rank"] * 1
    + df["PRED_rank"] * 4  # 미래예측 가중치 강화
)

df = df.sort_values("score")

print("\n=== 하이브리드 퀀트 점수표 ===")
print(df)

# ========= 8. 상위 20% 종목 선택 ============
top_n = max(1, int(len(df) * 0.2))
portfolio = df.index[:top_n]
print("\n선정된 포트폴리오:", list(portfolio))

# ========= 9. 전략 수익률 계산 ============
portfolio_price = price[portfolio].mean(axis=1)
strategy_return = portfolio_price.iloc[-1] / portfolio_price.iloc[0] - 1

print("\n전략 수익률: {:.2f}%".format(strategy_return * 100))

# ========= 10. 벤치마크 ============
sp500 = yf.download("^GSPC", start="2015-01-01")["Close"]
kospi = yf.download("069500.KS", start="2015-01-01")["Close"]

sp500_ret = float(sp500.iloc[-1]) / float(sp500.iloc[0]) - 1
kospi_ret = float(kospi.iloc[-1]) / float(kospi.iloc[0]) - 1

print("S&P500 수익률: {:.2f}%".format(sp500_ret * 100))
print("KOSPI200 수익률: {:.2f}%".format(kospi_ret * 100))

# ========= 11. 그래프 ============
plt.figure(figsize=(12,6))
plt.plot(portfolio_price / portfolio_price.iloc[0], label="하이브리드 전략")
plt.plot(sp500 / sp500.iloc[0], label="S&P500")
plt.plot(kospi / kospi.iloc[0], label="KOSPI200")
plt.title("하이브리드 퀀트 전략 vs 시장지수")
plt.legend()
plt.show()