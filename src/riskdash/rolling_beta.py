"""Rolling OLS beta of asset returns vs a market/factor series.

Windows are trailing and inclusive of the current bar only — no look-ahead.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series]


def rolling_beta(
    asset_returns: ArrayLike,
    market_returns: ArrayLike,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing OLS slope of asset ~ market over window bars.

    At index t the regression uses observations [t-window+1, t] only.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if min_periods is None:
        min_periods = window
    if min_periods < 2:
        raise ValueError("min_periods must be >= 2")

    y = pd.Series(np.asarray(asset_returns, dtype=float).ravel(), name="asset")
    x = pd.Series(np.asarray(market_returns, dtype=float).ravel(), name="market")
    if len(y) != len(x):
        raise ValueError("asset_returns and market_returns must have equal length")

    idx = pd.RangeIndex(len(y))
    y.index = idx
    x.index = idx

    def _ols_beta(i: int) -> float:
        start = max(0, i - window + 1)
        yy = y.iloc[start : i + 1].to_numpy()
        xx = x.iloc[start : i + 1].to_numpy()
        mask = np.isfinite(yy) & np.isfinite(xx)
        yy, xx = yy[mask], xx[mask]
        if yy.size < min_periods:
            return float("nan")
        x_mean = xx.mean()
        y_mean = yy.mean()
        var_x = np.sum((xx - x_mean) ** 2)
        if var_x < 1e-18:
            return float("nan")
        cov = np.sum((xx - x_mean) * (yy - y_mean))
        return float(cov / var_x)

    betas = [_ols_beta(i) for i in range(len(y))]
    return pd.Series(betas, index=idx, name="beta")


def rolling_beta_last(
    asset_returns: ArrayLike,
    market_returns: ArrayLike,
    window: int = 60,
) -> float:
    """Final (most recent) trailing beta scalar."""
    series = rolling_beta(asset_returns, market_returns, window=window)
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.iloc[-1])
