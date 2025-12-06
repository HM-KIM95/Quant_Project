import requests
import pandas as pd
import yfinance as yf

# -------------------------------
# ⭐ 1. 한국 종목 스크리닝
# -------------------------------
def get_korean_stock_list():
    from pykrx import stock

    kospi = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")

    tickers = kospi + kosdaq
    rows = []

    from pykrx import stock
    for t in tickers:
        name = stock.get_market_ticker_name(t)
        rows.append({"회사명": name, "종목코드": t + ".KS"})

    return pd.DataFrame(rows)

def get_korean_fundamental(ticker):
    from pykrx import stock
    code = ticker.replace(".KS", "")

    try:
        df = stock.get_market_fundamental("20230101", "20231231", code)
        recent = df.iloc[-1]

        per = recent["PER"]
        pbr = recent["PBR"]
        roe = recent["ROE"]

        return per, pbr, roe
    except:
        return None, None, None


# -------------------------------
# ⭐ 2. 미국 종목 스크리닝
# -------------------------------
def get_us_fundamental(ticker):
    """
    yfinance에서 PER, PBR, ROE를 가져옵니다.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    try:
        per = info.get("trailingPE", None)
        pbr = info.get("priceToBook", None)
        roe = info.get("returnOnEquity", None)
        if roe is not None:
            roe = roe * 100  # % 변환
        return per, pbr, roe
    except:
        return None, None, None


# -------------------------------
# ⭐ 3. 필터 조건
# -------------------------------
def pass_filter(per, pbr, roe):
    if per is None or pbr is None or roe is None:
        return False
    return (per <= 15) and (pbr <= 1.5) and (roe >= 15)


# -------------------------------
# ⭐ 4. 실행 메인 함수
# -------------------------------
def main():
    kr_list = get_korean_stock_list()
    us_list = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK-B"]  # 샘플 리스트

    results = []

    print("📌 한국 종목 스크리닝 중...")
    for _, row in kr_list.iterrows():
        name, ticker = row["회사명"], row["종목코드"]
        per, pbr, roe = get_korean_fundamental(ticker)
        if pass_filter(per, pbr, roe):
            results.append([ticker, name, per, pbr, roe])

    print("📌 미국 종목 스크리닝 중...")
    for ticker in us_list:
        per, pbr, roe = get_us_fundamental(ticker)
        if pass_filter(per, pbr, roe):
            results.append([ticker, ticker, per, pbr, roe])

    df = pd.DataFrame(results, columns=["Ticker", "Name", "PER", "PBR", "ROE"])
    print("\n🎯 조건 만족 종목 리스트")
    if df.empty:
        print("⚠️ 조건을 만족하는 종목이 없습니다.")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()