"""Cost helpers: net equity drag widens drawdowns vs gross."""

from __future__ import annotations

import numpy as np
import pytest

from riskdash.costs import (
    apply_costs,
    cost_drag_summary,
    equity_curve,
    net_of_cost_returns,
)
from riskdash.metrics import max_drawdown


def test_apply_costs_constant_turnover():
    r = np.array([0.01, 0.01, 0.01], dtype=float)
    net = apply_costs(r, positions=None, cost_bps=10.0)
    assert net == pytest.approx(r - 0.001)


def test_position_turnover_costs():
    r = np.array([0.01, 0.01, 0.01], dtype=float)
    pos = np.array([1.0, 0.0, 1.0], dtype=float)
    net = apply_costs(r, positions=pos, cost_bps=10.0)
    assert net == pytest.approx(r - 0.001)


def test_costs_widen_drawdowns():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.015, size=300)
    drag = cost_drag_summary(r, positions=None, cost_bps=10.0)
    assert drag["net_max_drawdown"] <= drag["gross_max_drawdown"] + 1e-12
    assert drag["drawdown_widening"] <= 1e-12
    assert drag["net_terminal_equity"] < drag["gross_terminal_equity"]
    assert max_drawdown(net_of_cost_returns(r, cost_bps=10.0)) <= max_drawdown(r) + 1e-12


def test_equity_curve():
    r = np.array([0.1, -0.1], dtype=float)
    eq = equity_curve(r, start=1.0)
    assert eq[0] == pytest.approx(1.1)
    assert eq[1] == pytest.approx(1.1 * 0.9)
