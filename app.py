import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime as dt

from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression, Ridge

st.set_page_config(page_title="AutoQuantView", layout="wide")
st.title("AutoQuantView (SQL + Regression + Automations)")

WATCHLIST = {
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

engine = create_engine("sqlite:///data/autoquantview.db", future=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT NOT NULL,
                date   TEXT NOT NULL,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS automation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                payload_json TEXT
            );
        """))

def upsert_prices(ticker: str, df: pd.DataFrame):
    df2 = df.copy()
    df2["ticker"] = ticker
    df2["date"] = pd.to_datetime(df2["Date"]).dt.date.astype(str)
    df2 = df2[["ticker","date","Open","High","Low","Close","Volume"]]
    df2.columns = ["ticker","date","open","high","low","close","volume"]
    rows = df2.to_dict(orient="records")
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume)
            VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
        """), rows)

def read_prices(ticker: str, start_date: pd.Timestamp | None = None) -> pd.DataFrame:
    q = "SELECT date, close FROM prices WHERE ticker = :t"
    params = {"t": ticker}
    if start_date is not None:
        q += " AND date >= :sd"
        params["sd"] = start_date.date().isoformat()
    q += " ORDER BY date ASC"
    with engine.begin() as conn:
        rows = conn.execute(text(q), params).fetchall()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows, columns=["date","close"])
    out["date"] = pd.to_datetime(out["date"])
    return out

@st.cache_data(ttl=60)
def load_from_db_or_fetch(ticker: str, period: str) -> pd.DataFrame:
    period_map = {"1mo": 35, "3mo": 110, "6mo": 220, "1y": 380, "2y": 760, "5y": 1900}
    days = period_map.get(period, 380)
    start_date = pd.Timestamp.today() - pd.Timedelta(days=days)

    df_db = read_prices(ticker, start_date=start_date)
    if not df_db.empty:
        return df_db

    df_yf = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False).reset_index()
    if df_yf.empty:
        return pd.DataFrame()

    upsert_prices(ticker, df_yf)
    return read_prices(ticker, start_date=start_date)

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

    return df, model, y_hat, resid, resid_std

def forecast(df_fitted: pd.DataFrame, model, horizon_days: int, resid_std: float):
    last_t = df_fitted["t"].iloc[-1]
    future_t = np.arange(last_t + 1, last_t + 1 + horizon_days).reshape(-1, 1)
    future_pred = model.predict(future_t)
    future_dates = next_business_days(df_fitted["date"].iloc[-1], horizon_days)

    z = 1.96
    lower = future_pred - z * resid_std
    upper = future_pred + z * resid_std
    return future_dates, future_pred, lower, upper

def log_event(ticker: str, event_type: str, severity: str, message: str, payload: dict | None = None):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO automation_events (created_at, ticker, event_type, severity, message, payload_json)
            VALUES (:created_at, :ticker, :event_type, :severity, :message, :payload_json)
        """), {
            "created_at": dt.datetime.utcnow().isoformat(),
            "ticker": ticker,
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "payload_json": json.dumps(payload) if payload else None
        })

def recent_events(ticker: str, limit: int = 20) -> pd.DataFrame:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT created_at, event_type, severity, message
            FROM automation_events
            WHERE ticker = :t
            ORDER BY id DESC
            LIMIT :lim
        """), {"t": ticker, "lim": limit}).fetchall()
    if not rows:
        return pd.DataFrame(columns=["created_at","event_type","severity","message"])
    return pd.DataFrame(rows, columns=["created_at","event_type","severity","message"])

def compute_automations(ticker: str, df: pd.DataFrame, df_fitted: pd.DataFrame, y_hat: np.ndarray):
    """
    Automation checks (simple + interview-friendly):
    1) Deviation: actual deviates from fitted trend today beyond threshold
    2) Volatility: 20d rolling vol too high
    3) Model drift: recent MAE worse than earlier MAE
    """
    messages = []

    if len(df_fitted) < 30:
        return messages

    # --- 1) Deviation alert (actual vs model fit on latest day)
    latest_actual = float(df_fitted["close"].iloc[-1])
    latest_pred = float(y_hat[-1])
    deviation_pct = (latest_actual - latest_pred) / max(1e-9, latest_pred)

    dev_thresh = st.session_state.get("dev_thresh", 0.05)
    if abs(deviation_pct) >= dev_thresh:
        sev = "HIGH" if abs(deviation_pct) >= dev_thresh * 1.8 else "MEDIUM"
        msg = f"Deviation alert: latest close deviates from model trend by {deviation_pct*100:.2f}%."
        log_event(ticker, "DEVIATION", sev, msg, {"latest_actual": latest_actual, "latest_pred": latest_pred, "deviation_pct": deviation_pct})
        messages.append((sev, msg))

    # --- 2) Volatility alert (rolling 20d vol on returns)
    df_tmp = df_fitted.copy()
    df_tmp["ret_1d"] = df_tmp["close"].pct_change()
    df_tmp["vol_20d"] = df_tmp["ret_1d"].rolling(20).std()
    vol_20 = float(df_tmp["vol_20d"].iloc[-1]) if not np.isnan(df_tmp["vol_20d"].iloc[-1]) else 0.0

    vol_thresh = st.session_state.get("vol_thresh", 0.03)
    if vol_20 >= vol_thresh:
        sev = "HIGH" if vol_20 >= vol_thresh * 1.5 else "MEDIUM"
        msg = f"Volatility alert: 20-day volatility is {vol_20:.4f} (threshold {vol_thresh:.4f})."
        log_event(ticker, "VOLATILITY", sev, msg, {"vol_20d": vol_20, "threshold": vol_thresh})
        messages.append((sev, msg))

    # --- 3) Model drift alert (rolling MAE comparison)
    abs_err = np.abs(df_fitted["close"].values - y_hat)
    # Compare last 20 days MAE vs previous 60 days MAE
    if len(abs_err) >= 90:
        recent_mae = float(np.mean(abs_err[-20:]))
        prev_mae = float(np.mean(abs_err[-80:-20]))
        drift_ratio = (recent_mae / max(1e-9, prev_mae))

        drift_thresh = st.session_state.get("drift_thresh", 1.25)
        if drift_ratio >= drift_thresh:
            sev = "HIGH" if drift_ratio >= drift_thresh * 1.3 else "MEDIUM"
            msg = f"Model drift alert: recent MAE is {drift_ratio:.2f}× worse than prior period."
            log_event(ticker, "MODEL_DRIFT", sev, msg, {"recent_mae": recent_mae, "prev_mae": prev_mae, "ratio": drift_ratio})
            messages.append((sev, msg))

    return messages

def daily_brief(ticker: str, selected_label: str, latest_close: float, resid_std: float, automation_msgs: list):
    # Simple free “AI brief” (templated summarization)
    lines = []
    lines.append(f"Daily Brief for {selected_label} ({ticker})")
    lines.append(f"- Latest close: ₹{latest_close:,.2f}")
    lines.append(f"- Model residual σ (fit noise proxy): {resid_std:,.2f}")

    if not automation_msgs:
        lines.append("- No alerts triggered by the automation rules today.")
        lines.append("- Interpretation: trend and volatility look within your configured thresholds.")
    else:
        # sort high first
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        automation_msgs = sorted(automation_msgs, key=lambda x: order.get(x[0], 9))
        lines.append(f"- Alerts triggered: {len(automation_msgs)}")
        for sev, msg in automation_msgs[:5]:
            lines.append(f"  - [{sev}] {msg}")

        lines.append("- Suggested next step: review the forecast band and recent volatility before making any decisions.")

    return "\n".join(lines)

# Initialize DB
init_db()

# --- UI controls
top1, top2, top3 = st.columns([2, 1, 1])
with top1:
    selected_label = st.selectbox("Select company", list(WATCHLIST.keys()), index=0)
with top2:
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
with top3:
    model_type = st.selectbox("Model", ["Linear", "Ridge"], index=1)

horizon = st.slider("Forecast horizon (business days)", min_value=7, max_value=60, value=14, step=1)

with st.sidebar:
    st.header("Automation thresholds")
    dev_thresh = st.slider("Deviation threshold (%)", 1, 20, 5, 1) / 100.0
    vol_thresh = st.slider("Volatility threshold (20d)", 1, 10, 3, 1) / 100.0
    drift_thresh = st.slider("Model drift threshold (ratio)", 110, 250, 125, 5) / 100.0

    st.session_state["dev_thresh"] = dev_thresh
    st.session_state["vol_thresh"] = vol_thresh
    st.session_state["drift_thresh"] = drift_thresh

ticker = WATCHLIST[selected_label]
df = load_from_db_or_fetch(ticker, period)

if df.empty:
    st.error("No data returned. Try another ticker or check your connection.")
    st.stop()

# Fit + forecast
df_fitted, model, y_hat, resid, resid_std = fit_model(df, model_type)
future_dates, future_pred, lower, upper = forecast(df_fitted, model, horizon, resid_std)

latest_date = df_fitted["date"].max().date()
latest_close = float(df_fitted.loc[df_fitted["date"] == df_fitted["date"].max(), "close"].iloc[0])

st.caption(
    f"Ticker: {ticker} | Latest: {latest_date} | Close: ₹{latest_close:,.2f} | "
    f"Rows: {len(df_fitted)} | Residual σ: {resid_std:,.2f}"
)

# Run automations (logs events)
automation_msgs = compute_automations(ticker, df, df_fitted, y_hat)

# Daily brief
st.subheader("Daily Brief")
st.code(daily_brief(ticker, selected_label, latest_close, resid_std, automation_msgs))

# Plot (small)
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

with st.expander("Recent automation events (stored in SQL)"):
    ev = recent_events(ticker, limit=30)
    st.dataframe(ev, use_container_width=True)

with st.expander("Forecast table"):
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": future_pred,
        "lower_95": lower,
        "upper_95": upper
    })
    st.dataframe(forecast_df, use_container_width=True)
