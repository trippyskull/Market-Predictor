import pandas as pd
from sqlalchemy import create_engine, text

# Single shared engine for the app (SQLite file)
engine = create_engine("sqlite:///data/autoquantview.db", future=True)

def init_db():
    """Create tables if they don't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT NOT NULL,
                date   TEXT NOT NULL,
                close  REAL,
                PRIMARY KEY (ticker, date)
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS mf_nav (
                scheme_code TEXT NOT NULL,
                nav_date TEXT NOT NULL,
                nav REAL,
                scheme_name TEXT,
                PRIMARY KEY (scheme_code, nav_date)
            );
        """))

def upsert_stock_prices(ticker: str, df_yf: pd.DataFrame):
    """
    Upsert stock closes into SQL.
    Expected yfinance df: columns include Date, Close
    """
    df2 = df_yf.copy()
    df2["ticker"] = ticker
    df2["date"] = pd.to_datetime(df2["Date"]).dt.date.astype(str)
    df2 = df2[["ticker", "date", "Close"]]
    df2.columns = ["ticker", "date", "close"]
    rows = df2.to_dict(orient="records")
    if not rows:
        return

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO prices (ticker, date, close)
            VALUES (:ticker, :date, :close)
        """), rows)

def read_stock_prices(ticker: str, start_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Read stock close series from SQL. Returns df with columns: date, close
    """
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

    out = pd.DataFrame(rows, columns=["date", "close"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    return out

def upsert_mf_nav(df_nav: pd.DataFrame):
    """
    Upsert mutual fund NAV history into SQL.
    Expected df columns: scheme_code, scheme_name, nav_date (datetime), nav (float)
    """
    if df_nav.empty:
        return

    df2 = df_nav.copy()
    df2["nav_date"] = pd.to_datetime(df2["nav_date"], errors="coerce").dt.date.astype(str)
    df2["nav"] = pd.to_numeric(df2["nav"], errors="coerce")
    df2 = df2.dropna(subset=["scheme_code", "nav_date", "nav"])

    rows = df2.to_dict(orient="records")
    if not rows:
        return

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO mf_nav (scheme_code, nav_date, nav, scheme_name)
            VALUES (:scheme_code, :nav_date, :nav, :scheme_name)
        """), rows)

def read_mf_nav(scheme_code: str) -> pd.DataFrame:
    """
    Read mutual fund NAV series from SQL.
    Returns df columns: nav_date, nav, scheme_name
    """
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT nav_date, nav, scheme_name
            FROM mf_nav
            WHERE scheme_code = :c
            ORDER BY nav_date ASC
        """), {"c": scheme_code}).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["nav_date", "nav", "scheme_name"])
    df["nav_date"] = pd.to_datetime(df["nav_date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["nav_date", "nav"]).sort_values("nav_date")
    return df
