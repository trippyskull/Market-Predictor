import numpy as np
import pandas as pd

from core.alpha_lab import compute_features, evaluate_alpha_expression, backtest_market_neutral, performance_summary


def sample_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=80)
    a = np.linspace(100, 140, len(idx))
    b = np.linspace(120, 90, len(idx))
    c = np.linspace(80, 100, len(idx))
    d = np.linspace(60, 110, len(idx))
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=idx)


def test_expression_evaluates_dataframe():
    prices = sample_prices()
    features = compute_features(prices)
    alpha = evaluate_alpha_expression("zscore(ts_mean(ret_1d, 5))", features)
    assert isinstance(alpha, pd.DataFrame)
    assert list(alpha.columns) == ["A", "B", "C", "D"]


def test_backtest_and_summary_has_metrics():
    prices = sample_prices()
    features = compute_features(prices)
    alpha = evaluate_alpha_expression("zscore(ret_1d)", features)
    pnl, _ = backtest_market_neutral(alpha, features["ret_1d"])
    stats = performance_summary(pnl)

    assert stats["days"] >= 10
    assert "sharpe" in stats
    assert "max_drawdown" in stats
