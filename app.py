import streamlit as st

from core.db import init_db

from tabs.stocks_tab import render_stocks_tab
from tabs.mutual_funds_tab import render_mutual_funds_tab
from tabs.portfolio_tab import render_portfolio_tab
from tabs.playground_tab import render_playground_tab


st.set_page_config(page_title="Market Predictor", layout="wide")
st.title("Market Predictor")

# Initialize database once
init_db()

tab_stocks, tab_mf, tab_portfolio, tab_playground = st.tabs(
    ["Stocks", "Mutual Funds", "Portfolio", "Playground"]
)

with tab_stocks:
    render_stocks_tab()

with tab_mf:
    render_mutual_funds_tab()

with tab_portfolio:
    render_portfolio_tab()

with tab_playground:
    render_playground_tab()
