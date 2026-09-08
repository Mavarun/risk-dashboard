"""Synthetic data shape and report builder smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from riskdash.data import load_returns, make_synthetic_factor_portfolio
from riskdash.report import build_report, metrics_table


def test_synthetic_shape_and_seed():
    a = make_synthetic_factor_portfolio(n=100, seed=42)
    b = make_synthetic_factor_portfolio(n=100, seed=42)
    assert list(a.columns) == ["date", "market", "portfolio", "position"]
    assert len(a) == 100
    assert np.allclose(a["portfolio"], b["portfolio"])


def test_load_returns_synthetic():
    panel = load_returns(source="synthetic", n=50, seed=1)
    assert len(panel) == 50


def test_build_report_keys():
    panel = make_synthetic_factor_portfolio(n=200, seed=5, true_beta=1.2)
    report = build_report(panel, window=40, cost_bps=5.0)
    assert "trailing_beta" in report
    assert abs(report["trailing_beta"] - 1.2) < 0.15
    table = metrics_table(report)
    assert set(table["series"]) == {"gross_portfolio", "net_portfolio", "market"}
    assert report["cost_drag"]["net_max_drawdown"] <= report["cost_drag"]["gross_max_drawdown"] + 1e-12
