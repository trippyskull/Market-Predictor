import streamlit as st
import matplotlib.pyplot as plt

from core.data_sources import load_amfi_scheme_list, fetch_mfapi_nav_history
from core.db import read_mf_nav, upsert_mf_nav
from core.utils import filter_by_period, simple_return


def render_mutual_funds_tab():
    st.subheader("Mutual Fund NAV Tracker")

    scheme_df = load_amfi_scheme_list()
    if scheme_df.empty:
        st.error("Could not load AMFI scheme list right now. Please retry later.")
        st.stop()

    query = st.text_input("Search your mutual fund name", value="ICICI Prudential")
    matches = scheme_df[scheme_df["scheme_name"].str.contains(query, case=False, na=False)].head(30)

    if matches.empty:
        st.info("No matches. Try: Parag Parikh, Axis Midcap, HDFC Small, SBI Gold, Motilal Oswal, etc.")
        st.stop()

    pick = st.selectbox(
        "Select scheme (from search results)",
        matches.apply(lambda r: f"{r['scheme_name']}  |  {r['scheme_code']}", axis=1).tolist()
    )
    scheme_code = pick.split("|")[-1].strip()

    colA, colB = st.columns([1, 1])
    with colA:
        st.write("Scheme code:", scheme_code)
    with colB:
        if st.button("Fetch/Refresh NAV history"):
            df_nav_new = fetch_mfapi_nav_history(scheme_code)
            if df_nav_new.empty:
                st.error("Could not fetch NAV history for this scheme.")
            else:
                upsert_mf_nav(df_nav_new)
                st.success(f"Saved {len(df_nav_new)} NAV rows to SQL.")

    df_nav_all = read_mf_nav(scheme_code)
    if df_nav_all.empty:
        st.warning("No NAV data in SQL yet. Click 'Fetch/Refresh NAV history'.")
        st.stop()

    mf_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
    df_nav = filter_by_period(df_nav_all, "nav_date", mf_period)

    st.caption(
        f"NAV rows shown: {len(df_nav)} | "
        f"Range: {df_nav['nav_date'].min().date()} → {df_nav['nav_date'].max().date()}"
    )

    scheme_name = df_nav_all["scheme_name"].iloc[-1]
    latest_nav = float(df_nav_all["nav"].iloc[-1])
    latest_nav_date = df_nav_all["nav_date"].iloc[-1].date()

    st.caption(f"{scheme_name} | Latest NAV: ₹{latest_nav:,.4f} | Date: {latest_nav_date}")

    r = simple_return(df_nav, "nav_date", "nav")
    st.write("Return over selected period:", "NA" if r is None else f"{r*100:.2f}%")

    fig2, ax2 = plt.subplots(figsize=(6.5, 3.2))
    ax2.plot(df_nav["nav_date"], df_nav["nav"])
    ax2.set_title("NAV over time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("NAV (INR)")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=False)

    with st.expander("Show NAV data (from SQL)"):
        st.dataframe(df_nav.tail(120), use_container_width=True)
