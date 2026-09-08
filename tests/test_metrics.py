"""Known-formula checks for Sharpe, Sortino, max DD, hit rate."""

from __future__ import annotations

import numpy as np
import pytest

from riskdash.metrics import hit_rate, max_drawdown, sharpe_ratio, sortino_ratio


def test_sharpe_known_path():
    r = np.array([0.01, 0.01, 0.01, 0.01], dtype=float)
    s = sharpe_ratio(r, risk_free=0.0, periods_per_year=1.0)
    assert s == 0.0 or np.isinf(s) or s > 0

    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, size=500)
    mu = x.mean()
    sd = x.std(ddof=1)
    expected = np.sqrt(252.0) * mu / sd
    assert sharpe_ratio(x, periods_per_year=252.0) == pytest.approx(expected, rel=1e-9)


def test_sortino_matches_hand_formula():
    r = np.array([0.02, -0.01, 0.015, -0.02, 0.01], dtype=float)
    target = 0.0
    excess_mean = r.mean()
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    expected = np.sqrt(252.0) * excess_mean / dd
    assert sortino_ratio(r, periods_per_year=252.0) == pytest.approx(expected, rel=1e-9)


def test_max_drawdown_hand_checked():
    r = np.array([0.10, -0.10, 0.10], dtype=float)
    assert max_drawdown(r) == pytest.approx(-0.1, rel=1e-9)
    r2 = np.array([0.05, 0.05, 0.05], dtype=float)
    assert max_drawdown(r2) == pytest.approx(0.0, abs=1e-12)


def test_hit_rate():
    r = np.array([0.01, -0.02, 0.03, 0.0, -0.01], dtype=float)
    assert hit_rate(r) == pytest.approx(2.0 / 5.0)
