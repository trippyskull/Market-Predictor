import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.alpha_lab import (
    DEFAULT_UNIVERSE,
    EXAMPLE_ALPHAS,
    compute_features,
    evaluate_alpha_expression,
    backtest_market_neutral,
    performance_summary,
)
from core.data_sources import yf_download_close
from core.db import read_stock_prices, upsert_stock_prices


@st.cache_data(ttl=60)
def load_universe_prices(universe: dict[str, str], period: str) -> pd.DataFrame:
    closes = []
    period_map = {"1mo": 35, "3mo": 110, "6mo": 220, "1y": 380, "2y": 760, "5y": 1900}
    days = period_map.get(period, 380)
    start_date = pd.Timestamp.today() - pd.Timedelta(days=days)

    for name, ticker in universe.items():
        df_db = read_stock_prices(ticker, start_date=start_date)
        if df_db.empty:
            df_yf = yf_download_close(ticker, period)
            if df_yf.empty:
                continue
            upsert_stock_prices(ticker, df_yf)
            df_db = read_stock_prices(ticker, start_date=start_date)

        if df_db.empty:
            continue

        s = df_db.set_index("date")["close"].rename(name)
        closes.append(s)

    if not closes:
        return pd.DataFrame()

    price_wide = pd.concat(closes, axis=1).sort_index()
    return price_wide.dropna(how="all")


def render_playground_tab():
    st.subheader("Alpha Playground (WorldQuant-style)")
    st.caption("Build and backtest cross-sectional alphas on Indian stocks.")

    c1, c2 = st.columns([1, 1])
    with c1:
        period = st.selectbox("Lookback period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    with c2:
        example = st.selectbox("Alpha template", ["Custom"] + list(EXAMPLE_ALPHAS.keys()), index=1)

    expression = EXAMPLE_ALPHAS.get(example, EXAMPLE_ALPHAS["Momentum 20D"])
    alpha_expr = st.text_input(
        "Alpha expression",
        value=expression,
        help=(
            "Available fields: ret_1d, ret_5d, ret_20d, mom_20d, vol_20d, ma_gap_20d, volume_z. "
            "Available functions: zscore, rank, ts_mean, ts_std, delay, delta, clip, abs."
        ),
    )

    price_wide = load_universe_prices(DEFAULT_UNIVERSE, period)
    if price_wide.empty or price_wide.shape[1] < 4:
        st.error("Not enough data available for the stock universe.")
        st.stop()

    features = compute_features(price_wide)
    try:
        alpha_signal = evaluate_alpha_expression(alpha_expr, features)
    except Exception as e:
        st.error(f"Invalid alpha expression: {e}")
        st.stop()

    pnl, weights = backtest_market_neutral(alpha_signal, features["ret_1d"])
    stats = performance_summary(pnl)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sharpe", f"{stats['sharpe']:.2f}")
    m2.metric("CAGR", f"{stats['cagr']*100:.2f}%")
    m3.metric("Max Drawdown", f"{stats['max_drawdown']*100:.2f}%")
    m4.metric("Total Return", f"{stats['total_return']*100:.2f}%")

    eq = (1 + pnl).cumprod()

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.plot(eq.index, eq.values)
    ax.set_title("Alpha strategy equity curve (market-neutral)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of ₹1")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

    latest_date = weights.dropna(how="all").index.max()
    if pd.notna(latest_date):
        latest_weights = weights.loc[latest_date].dropna().sort_values(ascending=False)
        st.write(f"Latest long/short weights as of {latest_date.date()}")
        wdf = pd.DataFrame({
            "stock": latest_weights.index,
            "weight": latest_weights.values,
        })
        st.dataframe(wdf, use_container_width=True)

    with st.expander("Universe prices"):
        st.dataframe(price_wide.tail(60), use_container_width=True)
