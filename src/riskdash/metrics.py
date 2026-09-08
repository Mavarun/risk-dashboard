"""Core risk metrics: Sharpe, Sortino, max drawdown, hit rate.

All functions operate on a return series already observed through time t.
None of these helpers peek at future returns.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, list]


def _as_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def sharpe_ratio(
    returns: ArrayLike,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized Sharpe ratio of a periodic return series."""
    r = _as_float_array(returns)
    if r.size < 2:
        return float("nan")
    excess = r - risk_free
    mu = float(np.mean(excess))
    sd = float(np.std(excess, ddof=1))
    if sd < 1e-15:
        return 0.0 if abs(mu) < 1e-15 else float("inf") * np.sign(mu)
    return float(np.sqrt(periods_per_year) * mu / sd)


def sortino_ratio(
    returns: ArrayLike,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
    target: float | None = None,
) -> float:
    """Annualized Sortino ratio using downside deviation below target."""
    r = _as_float_array(returns)
    if r.size < 2:
        return float("nan")
    if target is None:
        target = risk_free
    excess = r - risk_free
    downside = np.minimum(r - target, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    mu = float(np.mean(excess))
    if dd < 1e-15:
        return 0.0 if abs(mu) < 1e-15 else float("inf") * np.sign(mu)
    return float(np.sqrt(periods_per_year) * mu / dd)


def max_drawdown(returns: ArrayLike) -> float:
    """Max peak-to-trough drawdown of cumulative equity (non-positive)."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def hit_rate(returns: ArrayLike) -> float:
    """Fraction of periods with strictly positive returns."""
    r = _as_float_array(returns)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0.0))


def summarize_returns(
    returns: ArrayLike,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
    label: str = "portfolio",
) -> dict[str, float | str]:
    """Bundle core scalar metrics for reporting."""
    r = _as_float_array(returns)
    return {
        "label": label,
        "n": float(r.size),
        "mean": float(np.mean(r)) if r.size else float("nan"),
        "std": float(np.std(r, ddof=1)) if r.size > 1 else float("nan"),
        "sharpe": sharpe_ratio(r, risk_free=risk_free, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(r, risk_free=risk_free, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(r),
    }
