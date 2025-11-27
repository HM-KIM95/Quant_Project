import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='AppleGothic')
matplotlib.rcParams['axes.unicode_minus'] = False # 한글 폰트 추가
import requests
from bs4 import BeautifulSoup

# 1. 분석할 종목 리스트 정의 (삼성전자, 현대차)
tickers = ["005930.KS", "005380.KS", "058610.KS", "278470.KS", "034020.KS", "277810.KS", "087010.KS", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]

# 2. 가격 데이터 다운로드
data = yf.download(tickers, start="2015-01-01", auto_adjust=False)["Adj Close"]

# 3. 네이버 금융에서 PER, PBR 가져오기
def get_fundamental(ticker):
    # 한국 종목: 네이버 금융에서 PBR/PER 크롤링
    if ticker.endswith(".KS"):
        code = ticker.replace(".KS", "")
        url = f"https://finance.naver.com/item/main.nhn?code={code}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")
        try:
            per = soup.select_one("#_per").text.strip()
            pbr = soup.select_one("#_pbr").text.strip()
            per = float(per.replace(",", "")) if per not in ["N/A", "", None] else None
            pbr = float(pbr.replace(",", "")) if pbr not in ["N/A", "", None] else None
        except:
            per, pbr = None, None
        return per, pbr
    # 미국 종목: yfinance 재무 데이터 활용
    info = yf.Ticker(ticker).info
    per = info.get("trailingPE", None)
    pbr = info.get("priceToBook", None)
    return per, pbr

# 모든 종목에 대해 PER/PBR 수집
per = {}
pbr = {}
for t in tickers:
    per_val, pbr_val = get_fundamental(t)
    per[t] = per_val
    pbr[t] = pbr_val

# 5. 데이터프레임으로 정리 (정상적인 행/열 구조 생성)
df = pd.DataFrame.from_dict(
    {t: {"PER": per[t], "PBR": pbr[t]} for t in tickers},
    orient="index"
)

# None 값 대체
df["PER"] = df["PER"].fillna(9999)
df["PBR"] = df["PBR"].fillna(9999)

# 6. PER, PBR 순위 계산 (낮을수록 좋은 순위)
df["PER_rank"] = df["PER"].rank(ascending=True)
df["PBR_rank"] = df["PBR"].rank(ascending=True)

# 7. 종합 점수 (PER+PBR 낮은 순)
df["score"] = df["PER_rank"] + df["PBR_rank"]
df = df.sort_values("score")

print("\n=== 가치 기반 점수표 ===")
print(df)

# 8. 점수가 가장 낮은 종목 선택
best_stock = df.index[0]
print("\n선정된 최종 종목:", best_stock)

# 9. 선정된 종목의 가격 데이터 다운로드
price = yf.download(best_stock, start="2015-01-01", auto_adjust=False)["Adj Close"]

# 10. 수익률 계산
return_rate = price.iloc[-1] / price.iloc[0] - 1
print("\n전략 수익률: {:.2f}%".format(float(return_rate.iloc[0]) * 100))
print("\n 전략 수익률: {:.2f}%".format(float(return_rate) * 100 if not hasattr(return_rate, "iloc") else float(return_rate.iloc[0] * 100)))

# 11. KOSPI200 ETF 비교
benchmark = yf.download("069500.KS", start="2015-01-01", auto_adjust=False)["Adj Close"]
benchmark_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1
print("KOSPI ETF 수익률: {:.2f}%".format(float(benchmark_return) * 100 if not hasattr(benchmark_return, "iloc") else float(benchmark_return.iloc[0]) * 100))

# 12. 그래프 시각화
plt.figure(figsize=(12,6))
plt.plot(price / price.iloc[0], label=f"Strategy  ({best_stock})")
plt.plot(benchmark / benchmark.iloc[0], label="KOSPI200")
plt.legend()
plt.title("가치 전략 vs KOSPI200 비교 수익률")
plt.show()