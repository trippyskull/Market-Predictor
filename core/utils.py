import pandas as pd
from difflib import get_close_matches

def filter_by_period(df: pd.DataFrame, date_col: str, period: str) -> pd.DataFrame:
    """
    Filter df to last N months/years based on its latest date.
    Falls back to full df if filtered result is empty.
    period in: 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    """
    if df is None or df.empty:
        return df

    df = df.sort_values(date_col).copy()
    latest_dt = df[date_col].max()

    if period == "max":
        return df

    offsets = {
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "5y": pd.DateOffset(years=5),
    }
    cutoff = latest_dt - offsets[period]
    out = df[df[date_col] >= cutoff].copy()
    return out if not out.empty else df

def simple_return(df: pd.DataFrame, date_col: str, value_col: str) -> float | None:
    """
    Return over the window: last/first - 1.
    """
    if df is None or df.empty or len(df) < 2:
        return None
    d = df.sort_values(date_col)
    first = float(d[value_col].iloc[0])
    last = float(d[value_col].iloc[-1])
    if abs(first) < 1e-12:
        return None
    return (last / first) - 1.0

def best_name_match(query: str, candidates: list[str], cutoff: float = 0.55) -> str | None:
    """
    Fuzzy-match query against candidate strings.
    """
    if not candidates:
        return None
    match = get_close_matches(query, candidates, n=1, cutoff=cutoff)
    return match[0] if match else None
