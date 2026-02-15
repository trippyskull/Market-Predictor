import streamlit as st
import pandas as pd

from core.data_sources import load_amfi_scheme_list, fetch_mfapi_nav_history
from core.db import read_mf_nav, upsert_mf_nav
from core.utils import filter_by_period, simple_return, best_name_match


# Edit this list anytime (your portfolio)
MY_MF_PORTFOLIO = [
    "ICICI Prudential Gold ETF FoF Direct Growth",
    "Axis Midcap Direct Plan Growth",
    "Axis Global Equity Alpha FoF Direct Growth",
    "HDFC Dividend Yield Fund Direct Growth",
    "ICICI Prudential Multi Asset Fund Direct Growth",
    "HDFC Mid Cap Fund Direct Growth",
    "SBI Gold Direct Plan Growth",
    "Parag Parikh Flexi Cap Fund Direct Growth",
    "JM Flexicap Fund Direct Plan Growth",
    "Bandhan Value Fund Direct Growth",
    "Bandhan Small Cap Fund Direct Growth",
    "HDFC Small Cap Fund Direct Growth",
    "Motilal Oswal Midcap Fund Direct Growth",
]

def map_portfolio_names_to_scheme_codes(portfolio_names: list[str], scheme_df) -> list[dict]:
    """
    Map each portfolio fund name to the closest AMFI scheme name and code.
    Returns list of dicts:
      { portfolio_name, matched_scheme_name, scheme_code }
    """
    candidates = scheme_df["scheme_name"].tolist()
    out = []
    for pname in portfolio_names:
        matched = best_name_match(pname, candidates, cutoff=0.55)
        if not matched:
            out.append({"portfolio_name": pname, "matched_scheme_name": None, "scheme_code": None})
            continue
        row = scheme_df[scheme_df["scheme_name"] == matched].iloc[0]
        out.append({
            "portfolio_name": pname,
            "matched_scheme_name": matched,
            "scheme_code": str(row["scheme_code"])
        })
    return out

def render_portfolio_tab():
    st.subheader("My Mutual Fund Portfolio")

    scheme_df = load_amfi_scheme_list()
    if scheme_df.empty:
        st.error("Could not load AMFI scheme list right now. Please retry later.")
        st.stop()

    portfolio_period = st.selectbox("Portfolio Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

    st.write("Funds in portfolio:")
    st.code("\n".join(MY_MF_PORTFOLIO))

    mappings = map_portfolio_names_to_scheme_codes(MY_MF_PORTFOLIO, scheme_df)

    if st.button("Map funds + Fetch/Refresh NAV for ALL"):
        fetched = 0
        failed = 0
        for m in mappings:
            code = m["scheme_code"]
            if not code:
                failed += 1
                continue
            try:
                df_nav_new = fetch_mfapi_nav_history(code)
                if not df_nav_new.empty:
                    upsert_mf_nav(df_nav_new)
                    fetched += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        st.success(f"Done. Fetched/updated: {fetched}. Failed to map/fetch: {failed}.")

    # Build table
    rows = []
    for m in mappings:
        pname = m["portfolio_name"]
        matched = m["matched_scheme_name"]
        code = m["scheme_code"]

        if not code:
            rows.append({"Portfolio Name": pname, "Matched AMFI": None, "Scheme Code": None, "Rows": 0, "Return %": None})
            continue

        df_nav_all = read_mf_nav(code)
        if df_nav_all.empty:
            rows.append({"Portfolio Name": pname, "Matched AMFI": matched, "Scheme Code": code, "Rows": 0, "Return %": None})
            continue

        df_nav = filter_by_period(df_nav_all, "nav_date", portfolio_period)
        r = simple_return(df_nav, "nav_date", "nav")

        rows.append({
            "Portfolio Name": pname,
            "Matched AMFI": matched,
            "Scheme Code": code,
            "Rows": len(df_nav_all),
            "Return %": None if r is None else round(r * 100, 2),
        })

    port_df = pd.DataFrame(rows)
    st.dataframe(port_df, use_container_width=True)

    st.subheader("Ranking (by return %)")
    ranked = port_df.dropna(subset=["Return %"]).sort_values("Return %", ascending=False)
    if ranked.empty:
        st.info("No return data yet — click 'Map funds + Fetch/Refresh NAV for ALL'.")
    else:
        st.dataframe(ranked[["Portfolio Name", "Return %"]].head(15), use_container_width=True)
