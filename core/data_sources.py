import pandas as pd
import requests
import yfinance as yf
import streamlit as st

AMFI_NAVALL_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

@st.cache_data(ttl=60)
def yf_download_close(ticker: str, period: str) -> pd.DataFrame:
    """
    Download daily prices from yfinance.
    Returns df with columns: Date, Close
    """
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    ).reset_index()

    if df.empty:
        return pd.DataFrame()

    # Keep only what DB layer needs
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    return df[["Date", "Close"]].copy()

@st.cache_data(ttl=60*60*24)
def load_amfi_scheme_list() -> pd.DataFrame:
    """
    Build scheme_code + scheme_name list by parsing AMFI NAVAll.txt
    """
    try:
        r = requests.get(AMFI_NAVALL_URL, timeout=30)
        r.raise_for_status()
        text_data = r.text
    except Exception:
        return pd.DataFrame(columns=["scheme_code", "scheme_name"])

    rows = []
    for line in text_data.splitlines():
        if ";" not in line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 6:
            continue
        code = parts[0]
        name = parts[3]
        if code.isdigit():
            rows.append((code, name))

    return pd.DataFrame(rows, columns=["scheme_code", "scheme_name"]).drop_duplicates()

@st.cache_data(ttl=60*60*24)
def fetch_mfapi_nav_history(scheme_code: str) -> pd.DataFrame:
    """
    Fetch NAV history from mfapi.in
    Returns df columns: scheme_code, scheme_name, nav_date (datetime), nav (float)
    """
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()

    scheme_name = j.get("meta", {}).get("scheme_name", "")
    data = j.get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)  # columns: date, nav
    df["nav_date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["scheme_code"] = str(scheme_code)
    df["scheme_name"] = scheme_name

    df = df.dropna(subset=["nav_date", "nav"]).sort_values("nav_date")
    return df[["scheme_code", "scheme_name", "nav_date", "nav"]]
