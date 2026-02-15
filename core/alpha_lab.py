import numpy as np
import pandas as pd


DEFAULT_UNIVERSE = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "L&T": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "HCL Tech": "HCLTECH.NS",
    "Sun Pharma": "SUNPHARMA.NS",
}


EXAMPLE_ALPHAS = {
    "Momentum 20D": "zscore(ts_mean(ret_1d, 20))",
    "Short-term reversal": "-zscore(ret_1d)",
    "Low volatility": "-zscore(ts_std(ret_1d, 20))",
    "Momentum + volume": "zscore(ts_mean(ret_1d, 20)) + 0.5*zscore(volume_z)",
}


def compute_features(price_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ret_1d = price_wide.pct_change()
    return {
        "ret_1d": ret_1d,
        "ret_5d": price_wide.pct_change(5),
        "ret_20d": price_wide.pct_change(20),
        "mom_20d": price_wide / price_wide.shift(20) - 1,
        "vol_20d": ret_1d.rolling(20).std(),
        "ma_gap_20d": price_wide / price_wide.rolling(20).mean() - 1,
        "volume_z": ret_1d.rolling(20).std().rank(axis=1, pct=True),
    }


def ts_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).mean()


def ts_std(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).std()


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, np.nan)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def delay(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return df.shift(periods)


def delta(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return df - df.shift(periods)


def clip(df: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    return df.clip(lower=low, upper=high)


def evaluate_alpha_expression(expression: str, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        **features,
        "ts_mean": ts_mean,
        "ts_std": ts_std,
        "zscore": zscore,
        "rank": rank,
        "delay": delay,
        "delta": delta,
        "clip": clip,
        "abs": np.abs,
    }

    signal = eval(expression, safe_globals, safe_locals)
    if not isinstance(signal, pd.DataFrame):
        raise ValueError("Alpha expression must evaluate to a DataFrame signal.")

    return signal.replace([np.inf, -np.inf], np.nan)


def backtest_market_neutral(alpha: pd.DataFrame, daily_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    shifted_alpha = alpha.shift(1)
    demeaned = shifted_alpha.sub(shifted_alpha.mean(axis=1), axis=0)
    gross = demeaned.abs().sum(axis=1).replace(0, np.nan)
    weights = demeaned.div(gross, axis=0)
    pnl = (weights * daily_returns).sum(axis=1)
    return pnl.dropna(), weights


def performance_summary(pnl: pd.Series) -> dict[str, float]:
    if pnl.empty:
        return {
            "days": 0,
            "avg_daily_return": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }

    eq = (1 + pnl).cumprod()
    dd = eq / eq.cummax() - 1
    days = len(pnl)
    ann = 252
    avg = float(pnl.mean())
    vol = float(pnl.std())
    sharpe = (avg / vol) * np.sqrt(ann) if vol > 1e-12 else 0.0
    years = days / ann
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    return {
        "days": days,
        "avg_daily_return": avg,
        "volatility": vol,
        "sharpe": float(sharpe),
        "cagr": cagr,
        "max_drawdown": float(dd.min()),
        "total_return": float(eq.iloc[-1] - 1),
    }
