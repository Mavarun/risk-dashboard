"""Rolling beta: hand-checked OLS and no look-ahead in windows."""

from __future__ import annotations

import numpy as np
import pytest

from riskdash.rolling_beta import rolling_beta, rolling_beta_last


def test_rolling_beta_matches_ols():
    rng = np.random.default_rng(7)
    mkt = rng.normal(0, 0.01, size=200)
    true_b = 1.3
    asset = true_b * mkt + rng.normal(0, 0.002, size=200)
    window = 50
    betas = rolling_beta(asset, mkt, window=window)

    y = asset[-window:]
    x = mkt[-window:]
    x_c = x - x.mean()
    y_c = y - y.mean()
    hand = float(np.sum(x_c * y_c) / np.sum(x_c**2))
    assert betas.iloc[-1] == pytest.approx(hand, rel=1e-9)
    assert rolling_beta_last(asset, mkt, window=window) == pytest.approx(hand, rel=1e-9)
    assert abs(hand - true_b) < 0.05


def test_no_lookahead_in_rolling_windows():
    """Mutating future returns must not change beta at earlier indices."""
    rng = np.random.default_rng(11)
    mkt = rng.normal(0, 0.01, size=120)
    asset = 0.9 * mkt + rng.normal(0, 0.003, size=120)
    window = 30
    base = rolling_beta(asset, mkt, window=window).copy()

    asset_mut = asset.copy()
    asset_mut[-10:] = 999.0
    mut = rolling_beta(asset_mut, mkt, window=window)

    safe_end = len(asset) - 10 - 1
    assert np.allclose(
        base.iloc[: safe_end + 1].to_numpy(),
        mut.iloc[: safe_end + 1].to_numpy(),
        equal_nan=True,
    )
    assert not np.isclose(base.iloc[-1], mut.iloc[-1])


def test_min_periods_nan_prefix():
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.01, size=20)
    asset = 1.5 * mkt + rng.normal(0, 0.001, size=20)
    betas = rolling_beta(asset, mkt, window=10, min_periods=10)
    assert betas.iloc[:9].isna().all()
    assert np.isfinite(betas.iloc[9])
