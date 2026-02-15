import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge

from core.db import read_stock_prices, upsert_stock_prices
from core.data_sources import yf_download_close


STOCK_WATCHLIST = {
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "SBI (SBIN.NS)": "SBIN.NS",
    "ITC (ITC.NS)": "ITC.NS",
    "L&T (LT.NS)": "LT.NS",
    "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
}

@st.cache_data(ttl=60)
def load_stock_from_db_or_fetch(ticker: str, period: str) -> pd.DataFrame:
    period_map = {"1mo": 35, "3mo": 110, "6mo": 220, "1y": 380, "2y": 760, "5y": 1900}
    days = period_map.get(period, 380)
    start_date = pd.Timestamp.today() - pd.Timedelta(days=days)

    df_db = read_stock_prices(ticker, start_date=start_date)
    if not df_db.empty:
        return df_db

    df_yf = yf_download_close(ticker, period)
    if df_yf.empty:
        return pd.DataFrame()

    upsert_stock_prices(ticker, df_yf)
    return read_stock_prices(ticker, start_date=start_date)

def next_business_days(start: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start + pd.Timedelta(days=1), periods=n)

def fit_model(df: pd.DataFrame, model_type: str):
    df = df.sort_values("date").copy()
    df["t"] = np.arange(len(df), dtype=float)
    X = df[["t"]].values
    y = df["close"].values
    model = LinearRegression() if model_type == "Linear" else Ridge(alpha=1.0)
    model.fit(X, y)
    y_hat = model.predict(X)
    resid = y - y_hat
    resid_std = float(np.std(resid)) if len(resid) > 2 else 0.0
    return df, model, y_hat, resid_std

def forecast(df_fitted: pd.DataFrame, model, horizon_days: int, resid_std: float):
    last_t = df_fitted["t"].iloc[-1]
    future_t = np.arange(last_t + 1, last_t + 1 + horizon_days).reshape(-1, 1)
    future_pred = model.predict(future_t)
    future_dates = next_business_days(df_fitted["date"].iloc[-1], horizon_days)
    z = 1.96
    lower = future_pred - z * resid_std
    upper = future_pred + z * resid_std
    return future_dates, future_pred, lower, upper

def render_stocks_tab():
    st.subheader("Stocks")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        selected_label = st.selectbox("Select stock", list(STOCK_WATCHLIST.keys()), index=0)
    with c2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with c3:
        model_type = st.selectbox("Model", ["Linear", "Ridge"], index=1)

    horizon = st.slider("Forecast horizon (business days)", min_value=7, max_value=60, value=14, step=1)

    ticker = STOCK_WATCHLIST[selected_label]
    df = load_stock_from_db_or_fetch(ticker, period)

    if df.empty:
        st.error("No stock data returned.")
        st.stop()

    df_fitted, model, y_hat, resid_std = fit_model(df, model_type)
    future_dates, future_pred, lower, upper = forecast(df_fitted, model, horizon, resid_std)

    latest_date = df_fitted["date"].max().date()
    latest_close = float(df_fitted.loc[df_fitted["date"] == df_fitted["date"].max(), "close"].iloc[0])

    st.caption(f"{ticker} | Latest: {latest_date} | Close: ₹{latest_close:,.2f} | Rows: {len(df_fitted)}")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(df_fitted["date"], df_fitted["close"], label="Actual")
    ax.plot(future_dates, future_pred, linestyle="--", label="Forecast")
    ax.fill_between(future_dates, lower, upper, alpha=0.2, label="95% band")
    ax.set_title(f"{selected_label} — Actual + {model_type} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
